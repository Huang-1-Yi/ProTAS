"""
减少推理阶段不必要的计算开销
避免因文件结尾空行导致的数据丢失
避免结果保存的资源泄漏，确保异常安全
输出格式优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy             # 用于深度复制模型
import numpy as np
import tqdm             # 进度条
import pickle           # 序列化/反序列化

# Define the MultiStageModel class
class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, causal=False, use_graph=True, **graph_args):
        super().__init__()
        # 第一阶段模型（输入特征维度为dim）
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes, causal)
        # 后续阶段模型（输入为前一阶段的输出概率）
        self.stages = nn.ModuleList([
            copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes, causal))
            for _ in range(num_stages-1)
        ])
        self.use_graph = use_graph
        # 任务图学习模块
        if use_graph:
            self.graph_learner = TaskGraphLearner(**graph_args)

    def forward(self, x, mask):
        # 第一阶段前向传播
        out, out_app = self.stage1(x, mask)     # out: 分类结果, out_app: 进度预测
        
        # 存储各阶段输出
        outputs = out.unsqueeze(0)              # 增加维度0用于拼接
        outputs_app = out_app.unsqueeze(0)
        
        # 后续阶段处理
        for s in self.stages:
            # 使用前一阶段的softmax输出作为输入
            out, out_app = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
            outputs_app = torch.cat((outputs_app, out_app.unsqueeze(0)), dim=0)
        
        return outputs, outputs_app             # 所有阶段的输出

# Define the TaskGraphLearner class
class TaskGraphLearner(nn.Module):
    def __init__(self, init_graph_path, learnable=False, reg_weight=0.01, eta=0.01):
        super(TaskGraphLearner, self).__init__()
        
        # 加载预定义的任务图，使用with可以增强代码的可读性和异常处理
        with open(init_graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        
        # 转换为PyTorch参数
        matrix_pre, matrix_suc = self.graph['matrix_pre'], self.graph['matrix_suc']
        self.matrix_pre = nn.Parameter(torch.from_numpy(matrix_pre).float(), requires_grad=learnable)
        self.matrix_suc = nn.Parameter(torch.from_numpy(matrix_suc).float(), requires_grad=learnable)
        
        self.learnable = learnable
        if learnable:  # 保存原始图用于正则化
            self.matrix_pre_original = nn.Parameter(self.matrix_pre, requires_grad=False)
            self.matrix_suc_original = nn.Parameter(self.matrix_suc, requires_grad=False)

        self.reg_weight = reg_weight  # 正则化权重
        self.eta = eta  # 推理时调整参数

    def forward(self, cls, prg):
        # 分类概率
        action_prob = F.softmax(cls, dim=1)
        # 进度预测（限制在0-1范围）
        prg = torch.clamp(prg, min=0, max=1)# 每次推理都执行clamp
        # 计算完成状态（累积最大值）
        completion_status, _ = torch.cummax(prg, dim=-1)
        
        # 计算前驱和后继影响
        alpha_pre = torch.einsum('bkt,kK->bKt', 1 - completion_status, self.matrix_pre)
        alpha_suc = torch.einsum('bkt,kK->bKt', completion_status, self.matrix_suc)
        
        # 图损失（动作概率与图约束的乘积）
        graph_loss = ((alpha_pre + alpha_suc) * action_prob).mean()
        
        # 可学习时添加正则化
        if self.learnable:
            regularization = torch.mean((self.matrix_pre - self.matrix_pre_original) ** 2)
            return graph_loss + self.reg_weight * regularization
        
        return graph_loss

    def inference(self, cls, prg):
        # 与forward类似，但用于推理阶段
        # action_prob = F.softmax(cls, dim=1)# 计算分类概率，在推理阶段无用
        prg = torch.clamp(prg, min=0, max=1)
        completion_status, _ = torch.cummax(prg, dim=-1)
        
        alpha_pre = torch.einsum('bkt,kK->bKt', 1 - completion_status, self.matrix_pre)
        alpha_suc = torch.einsum('bkt,kK->bKt', completion_status, self.matrix_suc)
        
        # 调整分类logits（减去图约束）
        logits = cls - self.eta * (alpha_pre + alpha_suc)
        return logits

# Define the ProbabilityProgressFusionModel class
class ProbabilityProgressFusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # 1x1卷积融合分类和进度信息
        self.conv = nn.Conv1d(num_classes*2, num_classes, 1)

    def forward(self, in_cls, in_prg):
        ### in_cls: batch_size x num_classes x T
        ### in_prg: batch_size x num_classes x T
        # 拼接分类概率和进度预测 Concatenate classification and progress inputs
        input_concat = torch.cat((in_cls, in_prg), dim=1)
        out = self.conv(input_concat)
        return out


# Define the DilatedResidualLayer class
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, filter_size=3, causal=False):
        super(DilatedResidualLayer, self).__init__()
        self.causal = causal  # 是否因果卷积（用于实时处理）
        self.dilation = dilation
        
        # 计算填充大小
        padding = int(dilation * (filter_size-1) / 2)
        if causal:
            # 因果卷积（右侧填充）
            self.conv_dilated = nn.Conv1d(in_channels, out_channels, filter_size, 
                                         padding=padding*2, padding_mode='replicate', 
                                         dilation=dilation)
        else:
            # 标准空洞卷积
            self.conv_dilated = nn.Conv1d(in_channels, out_channels, filter_size,
                                         padding=padding, dilation=dilation)
        
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)  # 1x1卷积
        self.dropout = nn.Dropout()  # 防止过拟合

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        if self.causal:
            # 因果卷积需裁剪右侧多余填充
            out = out[..., :-self.dilation*2]
        
        out = self.conv_1x1(out)
        out = self.dropout(out)
        # 残差连接并应用掩码
        return (x + out) * mask[:, 0:1, :]

# Define the SingleStageModel class
class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, causal=False):
        super(SingleStageModel, self).__init__()
        # 特征维度转换
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        # 堆叠空洞残差层
        self.layers = nn.ModuleList([
            copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, causal=causal))
            for i in range(num_layers)
        ])
        # 分类输出层
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

        ### 动作进度预测模块 ###
        ### Action Progress Prediction (APP) module
        # GRU处理时序依赖
        self.gru_app = nn.GRU(num_f_maps, num_f_maps, num_layers=1, 
                             batch_first=True, bidirectional=not causal)
        # 进度预测输出层
        self.conv_app = nn.Conv1d(num_f_maps, num_classes, 1)
        # 概率-进度融合模块
        self.prob_fusion = ProbabilityProgressFusionModel(num_classes)

    def forward(self, x, mask):
        # 特征转换
        out = self.conv_1x1(x)
        # 通过残差层
        for layer in self.layers:
            out = layer(out, mask)
        
        # 分类分支
        prob_out = self.conv_out(out) * mask[:, 0:1, :]
        
        # 进度预测分支
        progress_out, _ = self.gru_app(out.permute(0, 2, 1))  # GRU需要(batch, seq, features)
        progress_out = progress_out.permute(0, 2, 1)  # 恢复为(batch, features, seq)
        progress_out = self.conv_app(progress_out) * mask[:, 0:1, :]
        
        # 融合分类和进度信息
        out = self.prob_fusion(prob_out, progress_out)
        out = out * mask[:, 0:1, :]  # 应用掩码
        
        return out, progress_out


# Define the Trainer class
class Trainer:
    def __init__(self, 
                 num_blocks, 
                 num_layers, 
                 num_f_maps, 
                 dim, 
                 num_classes, 
                 causal, 
                 logger, 
                 progress_lw=1,
                 use_graph=True, 
                 graph_lw=0.1, 
                 init_graph_path='', 
                 learnable=True
                 ):
        # 初始化模型
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes, causal, 
                     use_graph=use_graph, init_graph_path=init_graph_path, learnable=learnable)
        # 损失函数
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)  # 分类损失
        self.mse = nn.MSELoss(reduction='none')  # 进度预测损失
        self.num_classes = num_classes
        # 损失权重
        self.progress_lw = progress_lw  # 进度损失权重
        self.use_graph = use_graph
        self.graph_lw = graph_lw  # 图损失权重
        self.logger = logger  # 日志记录器

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_progress_loss = 0
            epoch_graph_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                # 获取批次数据
                batch_input, batch_target, batch_progress_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, batch_progress_target, mask = batch_input.to(device), batch_target.to(device), batch_progress_target.to(device), mask.to(device)
                
                optimizer.zero_grad()
                # 前向传播
                predictions, progress_predictions = self.model(batch_input, mask)

                loss = 0
                progress_loss = 0
                # 计算各阶段损失
                for p, progress_p in zip(predictions, progress_predictions):
                    # 分类损失
                    loss += self.ce(
                        p.transpose(2, 1).contiguous().view(-1, self.num_classes), 
                        batch_target.view(-1)
                        )
                    # 时序平滑正则化
                    loss += 0.15 * torch.mean(
                        torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16) * mask[:, :, 1:]
                        )
                    # 进度预测损失
                    progress_loss += self.mse(progress_p, batch_progress_target).mean()

                # 总损失
                loss += self.progress_lw * progress_loss
                epoch_progress_loss += self.progress_lw * progress_loss.item()
                
                # 图损失（如果使用图学习）
                graph_loss = self.model.graph_learner(predictions[-1], progress_predictions[-1])
                loss += self.graph_lw * graph_loss
                epoch_graph_loss += self.graph_lw * graph_loss.item()
                epoch_loss += loss.item()
                
                # 反向传播
                loss.backward()
                optimizer.step()

                # 计算准确率
                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            # 重置数据生成器
            batch_gen.reset()
            # 定期保存模型
            if (epoch + 1) % 5 == 0:
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            # 记录日志
            self.logger.info(f"[epoch {epoch+1}]: "
                            f"epoch loss = {epoch_loss/len(batch_gen.list_of_examples):.6f}, "
                            f"progress loss = {epoch_progress_loss/len(batch_gen.list_of_examples):.6f}, "
                            f"graph loss = {epoch_graph_loss/len(batch_gen.list_of_examples):.6f}, "
                            f"acc = {correct/total:.4f}")

    # 批量预测函数
    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate, feature_transpose=False, map_delimiter=' '):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            
            # 加载视频列表
            with open(vid_list_file, 'r') as f:
                list_of_vids = f.read().splitlines()# 从split('\n')[:-1]到splitlines()的转换是一个重要的改进，它使代码更健壮、更可靠，尤其是在处理来自不同系统的文件时

            for vid in list_of_vids:
                # 加载特征
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                if feature_transpose:
                    features = features.T
                features = features[:, ::sample_rate]  # 降采样

                # 准备输入
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                
                # 前向传播
                predictions, progress_predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                # 应用图推理
                final_predictions = self.model.graph_learner.inference(predictions[-1], progress_predictions[-1])
                _, predicted = torch.max(final_predictions.data, 1)

                # 压缩预测结果，转换为动作标签，
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    # 上采样恢复原始帧率
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                
                # 保存结果
                f_name = vid.split('/')[-1].split('.')[0]
                with open(f"{results_dir}/{f_name}", "w") as f_ptr:
                    f_ptr.write("### Frame level recognition: ###\n")
                    f_ptr.write(map_delimiter.join(recognition))

    # 在线预测函数（逐帧处理）
    def predict_online(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate, feature_transpose=False, map_delimiter=' '):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            
            # 加载视频列表
            with open(vid_list_file, 'r') as f:
                list_of_vids = f.read().splitlines()
            
            for vid in list_of_vids:
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                if feature_transpose:
                    features = features.T
                features = features[:, ::sample_rate]
                
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                n_frames = input_x.shape[-1]
                recognition = []
                
                # 逐帧处理
                for frame_i in tqdm.tqdm(range(n_frames)):
                    curr_input_x = input_x[:, :, :frame_i+1]  # 使用当前帧及之前所有帧
                    predictions, progress_predictions = self.model(curr_input_x, torch.ones(curr_input_x.size(), device=device))
                    final_predictions = self.model.graph_learner.inference(predictions[-1], progress_predictions[-1])
                    _, predicted = torch.max(final_predictions.data, 1)
                    predicted = predicted.squeeze(0)
                    
                    # 只取最后一帧的预测结果
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[-1].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                with open(f"{results_dir}/{f_name}", "w") as f_ptr:
                    f_ptr.write("### Frame level recognition: ###\n")
                    f_ptr.write(map_delimiter.join(recognition))
