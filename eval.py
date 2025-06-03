#!/usr/bin/python2.7
# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py

import numpy as np                                      # 导入数值计算库
import argparse                                         # 导入命令行参数解析库
from itertools import groupby                           # 导入分组连续相同元素的工具
import os                                               # 导入操作系统接口库
import json                                             # 导入JSON数据处理库

def read_file(path):                                    # 定义读取文件的函数
    with open(path, 'r') as f:                          # 以只读模式打开文件
        content = f.read()                              # 读取整个文件内容
        f.close()                                       # 关闭文件
    return content                                      # 返回文件内容

# 获取动作片段信息的函数
def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):  
    labels = []                                         # 存储动作标签
    starts = []                                         # 存储动作开始帧
    ends = []                                           # 存储动作结束帧
    last_label = frame_wise_labels[0]                   # 上一个标签初始化为第一帧
    
    if frame_wise_labels[0] not in bg_class:            # 如果第一帧不是背景类
        labels.append(frame_wise_labels[0])             # 添加标签
        starts.append(0)                                # 添加开始帧（0帧）
    
    for i in range(len(frame_wise_labels)):             # 遍历所有帧
        if frame_wise_labels[i] != last_label:          # 如果标签变化
            if frame_wise_labels[i] not in bg_class:    # 新动作开始（排除背景类）
                labels.append(frame_wise_labels[i])     # 添加新标签
                starts.append(i)                        # 添加新动作开始帧
            if last_label not in bg_class:              # 旧动作结束（排除背景类）
                ends.append(i)                          # 添加旧动作结束帧
            last_label = frame_wise_labels[i]           # 更新上一个标签
    
    if last_label not in bg_class:                      # 处理最后一帧
        ends.append(i + 1)                              # 添加最后一个动作的结束帧
    
    return labels, starts, ends                         # 返回动作信息

# Levenshtein距离计算函数
def levenstein(p, y, norm=False):                       
    m_row = len(p)                                      # 预测序列长度
    n_col = len(y)                                      # 真实序列长度
    D = np.zeros([m_row+1, n_col+1], np.float)          # 初始化编辑距离矩阵
    
    for i in range(m_row+1):                            # 初始化第一列（从p到空字符串的距离）
        D[i, 0] = i
    
    for i in range(n_col+1):                            # 初始化第一行（从空字符串到y的距离）
        D[0, i] = i
    
    for j in range(1, n_col+1):                         # 计算编辑距离
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:                        # 如果匹配
                D[i, j] = D[i-1, j-1]                   # 继承对角值
            else:                                       # 如果不匹配
                D[i, j] = min(D[i-1, j] + 1,            # 删除
                              D[i, j-1] + 1,            # 插入
                              D[i-1, j-1] + 1)          # 替换
    
    if norm:                                            # 如果需要归一化
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100 # 归一化编辑分数
    else:
        score = D[-1, -1]                               # 原始编辑距离
    
    return score                                        # 返回分数

# 编辑分数计算函数
def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):  
    # modified edit_score to remove consecutive duplicates after filtering out background
    recognized_no_bg = [a for a in recognized if a not in bg_class]         # 移除背景类（预测）
    ground_truth_no_bg = [a for a in ground_truth if a not in bg_class]     # 移除背景类（真实）
    P = [k for k, g in groupby(recognized_no_bg)]                           # 合并连续相同动作（预测）
    Y = [k for k, g in groupby(ground_truth_no_bg)]                         # 合并连续相同动作（真实）
    #P, _, _ = get_labels_start_end_time(recognized, bg_class)
    #Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)                                           # 返回归一化编辑分数


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):                # F分数计算函数
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)           # 获取预测动作片段
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)         # 获取真实动作片段

    tp = 0  # 真阳性计数
    fp = 0  # 假阳性计数
    hits = np.zeros(len(y_label))                       # 标记真实片段是否被匹配

    for j in range(len(p_label)):                       # 对每个预测片段进行评估
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)    # 计算交集
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)           # 计算并集
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])  # 计算IoU
        idx = np.array(IoU).argmax()                    # 获取IoU最大的真实片段索引

        if IoU[idx] >= overlap and not hits[idx]:       # 检查是否满足匹配条件
            tp += 1                                     # 真阳性增加
            hits[idx] = 1                               # 标记真实片段已被匹配
        else:
            fp += 1                                     # 假阳性增加

    fn = len(y_label) - sum(hits)                       # 计算假阴性（未被匹配的真实片段）
    return float(tp), float(fp), float(fn)              # 返回统计值


def evaluate(dataset, result_dir, split, exp_id, num_epochs):               # 主要评估函数
    ground_truth_path = "./data/"+dataset+"/groundTruth/"                   # 真实标签路径
    recog_path = result_dir                                                 # 预测结果路径 #"./results/"+exp_id+"/"+dataset+"/epoch"+str(num_epochs)+"/split_"+split+"/"
    file_list = "./data/"+dataset+"/splits/test.split"+split+".bundle"      # 测试文件列表
    list_of_videos = read_file(file_list).split('\n')[:-1]                  # 读取测试视频列表

    overlap = [.1, .25, .5]                                                 # IoU阈值设置
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)                      # 初始化统计数组

    correct = 0                                                             # 总正确帧数
    total = 0                                                               # 总帧数
    correct_wo_bg = 0                                                       # 忽略背景后的正确帧数
    total_wo_bg = 0                                                         # 忽略背景后的总帧数
    edit = 0                                                                # 总编辑分数

    map_delimiter = '|' if dataset in ['ptg', 'coffee', 'tea', 'pinwheels', 'oatmeal', 'quesadilla'] else ' '  # 数据集特定分隔符
    bg_class = ['BG'] if dataset in ['ptg', 'coffee', 'tea', 'pinwheels', 'oatmeal', 'quesadilla'] else ['background']  # 数据集特定背景类

    for vid in list_of_videos:                                              # 遍历每个测试视频
        if not vid.endswith('.txt'):                                        # 确保文件名以.txt结尾
            vid = vid + '.txt'
        gt_file = ground_truth_path + vid                                   # 真实标签文件路径
        gt_content = read_file(gt_file).split('\n')[0:-1]                   # 读取真实标签内容

        recog_file = os.path.join(recog_path, vid.split('.')[0])            # 预测结果文件路径
        recog_content = read_file(recog_file).split('\n')[1].split(map_delimiter)  # 读取预测结果内容
        
        for i in range(len(gt_content)):                                    # 计算帧级准确率
            if gt_content[i] not in bg_class:                               # 忽略背景后的统计
                total_wo_bg += 1
                if gt_content[i] == recog_content[i]:
                    correct_wo_bg += 1
            total += 1                                                      # 总体统计
            if gt_content[i] == recog_content[i]:
                correct += 1
        
        edit += edit_score(recog_content, gt_content, bg_class=bg_class)    # 累加编辑分数

        for s in range(len(overlap)):                                       # 计算F分数在不同IoU阈值下的统计
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s], bg_class)
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
            
    acc = 100*float(correct)/total                                          # 计算总体准确率
    acc_wo_bg = 100*float(correct_wo_bg)/total_wo_bg                        # 计算忽略背景准确率
    edit = (1.0*edit)/len(list_of_videos)                                   # 计算平均编辑分数
    res_list = [acc, acc_wo_bg, edit]                                       # 初始化结果列表

    #print("Acc: %.4f" % (100*float(correct)/total))
    #print('Edit: %.4f' % ((1.0*edit)/len(list_of_videos)))
    for s in range(len(overlap)):                                           # 计算不同IoU阈值下的F1分数
        precision = tp[s] / float(tp[s]+fp[s]) if (tp[s]+fp[s]) > 0 else 0  # 计算精确率
        recall = tp[s] / float(tp[s]+fn[s]) if (tp[s]+fn[s]) > 0 else 0     # 计算召回率
        f1 = 2.0 * (precision*recall) / (precision+recall) if (precision+recall) > 0 else 0  # 计算F1分数
        f1 = np.nan_to_num(f1)*100                                          # 处理NaN并转换为百分比
        #print('F1@%0.2f: %.4f' % (overlap[s], f1))
        res_list.append(f1)                                                 # 添加到结果列表
    
    # 打印最终结果
    print(exp_id, ' '.join(['{:.2f}'.format(r) for r in res_list]))         # 输出所有指标

    result_metrics = {                                                      # 构建结果字典
        'Acc': acc, 
        'Acc-bg': acc_wo_bg, 
        'Edit': edit, 
        'F1@10': res_list[-3], 
        'F1@25': res_list[-2], 
        'F1@50': res_list[-1]
    }
    
    result_path = os.path.join(recog_path, 'split'+split+'.eval.json')      # 结果文件路径
    with open(result_path, 'w') as fw:                                      # 保存结果为JSON文件
        json.dump(result_metrics, fw, indent=4)                             # 缩进格式保存

def main():
    parser = argparse.ArgumentParser()                              # 创建参数解析器

    parser.add_argument('--dataset', default="ptg")                 # 数据集名称参数
    parser.add_argument('--split', default='1')                     # 数据分割参数
    parser.add_argument('--exp_id', default='default', type=str)    # 实验ID参数
    parser.add_argument('--result_dir', default='', type=str)       # 结果目录参数
    parser.add_argument('--num_epochs', default=100, type=int)      # 训练轮数参数

    args = parser.parse_args()                                      # 解析参数
    evaluate(args.dataset, args.result_dir, args.split, args.exp_id, args.num_epochs)  # 调用评估函数

if __name__ == '__main__':
    main()
