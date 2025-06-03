#!/usr/bin/python2.7

import torch  # 导入PyTorch深度学习库
import numpy as np  # 导入NumPy数值计算库
import random  # 导入随机数生成模块
from collections import defaultdict  # 导入带默认值的字典

# 定义一个批处理生成器类
class BatchGenerator(object):
    # 初始化方法
    def __init__(self, num_classes, actions_dict, gt_path, features_path, progress_path, sample_rate, 
            feature_transpose=False):
        # 内部变量：存储所有样本ID的列表
        self.list_of_examples = list()
        # 内部变量：当前处理的样本索引
        self.index = 0
        # 输入参数：动作类别总数
        self.num_classes = num_classes
        # 输入参数：动作名称到索引的映射字典
        self.actions_dict = actions_dict
        # 输入参数：真实标签（ground truth）文件路径
        self.gt_path = gt_path
        # 输入参数：特征文件路径
        self.features_path = features_path
        # 输入参数：进度预测标签文件路径
        self.progress_path = progress_path
        # 输入参数：降采样率（用于减少时间分辨率）
        self.sample_rate = sample_rate
        # 输入参数：是否需要转置特征矩阵（特定数据集需要）
        self.feature_transpose = feature_transpose

    # 重置生成器方法
    def reset(self):
        # 重置索引到列表开始位置
        self.index = 0
        # 随机打乱样本顺序（用于每次epoch重新排序）
        random.shuffle(self.list_of_examples)

    # 检查是否有下一个批次
    def has_next(self):
        # 如果当前索引小于样本总数，返回True
        if self.index < len(self.list_of_examples):
            return True
        # 否则返回False
        return False

    # 读取视频列表文件
    def read_data(self, vid_list_file):
        # 打开视频列表文件
        file_ptr = open(vid_list_file, 'r')
        # 读取文件内容，按换行分割，并移除最后可能的空行
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        # 关闭文件
        file_ptr.close()
        # 随机打乱样本顺序
        random.shuffle(self.list_of_examples)

    # 获取下一个批次的方法
    def next_batch(self, batch_size):
        # 从当前索引获取一个批次的样本
        batch = self.list_of_examples[self.index:self.index + batch_size]
        # 更新索引以指向下一批的起点
        self.index += batch_size

        # 初始化批次数据存储列表
        batch_input = []  # 存储特征
        batch_target = []  # 存储动作标签
        batch_progress = []  # 存储进度标签
        
        # 遍历批次中的每个视频样本
        for vid in batch:
            # 加载特征文件（numpy格式）
            features = np.load(self.features_path + vid.split('.')[0] + '.npy')
            # 加载进度预测标签文件（numpy格式）
            progress_values = np.load(self.progress_path + vid.split('.')[0] + '.npy')
            
            # 如果需要，对特征矩阵进行转置
            if self.feature_transpose:
                features = features.T
            
            # 打开真实标签文件
            file_ptr = open(self.gt_path + vid, 'r')
            # 读取文件内容，按行分割并移除最后可能的空行
            content = file_ptr.read().split('\n')[:-1]
            # 创建标签数组，长度取特征维度和标签数的最小值（防止不匹配）
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            
            # 遍历每个时间步，将动作名称映射为索引
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            
            # 对特征进行降采样，并添加到批次
            batch_input.append(features[:, ::self.sample_rate])
            # 对动作标签进行降采样，并添加到批次
            batch_target.append(classes[::self.sample_rate])
            # 对进度标签进行降采样，并添加到批次
            batch_progress.append(progress_values[:, ::self.sample_rate])
        
        # 获取批次中各样本的实际长度（降采样后）
        length_of_sequences = list(map(len, batch_target))

        # 创建输入特征张量（初始化为0）
        # 维度: [batch_size, feature_dim, max_seq_len]
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        
        # 创建目标标签张量（初始化为-100，PyTorch中表示忽略索引）
        # 维度: [batch_size, max_seq_len]
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        
        # 创建进度标签张量（初始化为0）
        # 维度: [batch_size, num_classes, max_seq_len]
        batch_progress_tensor = torch.zeros(len(batch_input), np.shape(batch_progress[0])[0], max(length_of_sequences), dtype=torch.float)
        
        # 创建掩码张量（初始化为0）
        # 维度: [batch_size, num_classes, max_seq_len]
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        
        # 遍历批次中每个样本，填充张量
        for i in range(len(batch_input)):
            # 填充输入特征：当前样本对应位置填充实际特征
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            # 填充目标标签：当前样本对应位置填充实际标签
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            # 填充进度标签：当前样本对应位置填充实际进度
            batch_progress_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_progress[i])
            # 填充掩码：实际长度部分设为1（表示有效数据）
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        # 返回批次的四个张量
        return batch_input_tensor, batch_target_tensor, batch_progress_tensor, mask
