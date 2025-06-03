import numpy as np              # 导入数值计算库
import os                       # 导入操作系统接口库
from itertools import groupby   # 导入分组连续相同元素的工具
import tqdm                     # 导入进度条库

def write_progress_values(dataset, bg_class=[0], map_delimiter=' '):
    """
    Generate and write progress values for each action in the dataset.

    Parameters:
    - dataset (str): The name of the dataset.
    - bg_class (list): List of background class labels to ignore.
    - map_delimiter (str): Delimiter used in the mapping file.
    """
    # 设置文件路径
    gt_path = os.path.join('data', dataset, 'groundTruth')              # 真实标签路径
    mapping_file = os.path.join("data", dataset, "mapping.txt")         # 动作映射文件路径
    progress_path = os.path.join('data', dataset, 'progress')           # 进度值保存路径
    os.makedirs(progress_path, exist_ok=True)                           # 创建进度值目录（如果不存在）
    
    # 创建动作到索引的映射字典
    actions_dict = dict()                                               # 初始化动作字典
    with open(mapping_file, 'r') as f:                                  # 打开映射文件
        for line in f:                                                  # 遍历每一行
            actions = line.strip().split(map_delimiter)                 # 分割动作ID和名称
            actions_dict[actions[1]] = int(actions[0])                  # 添加动作名称到ID的映射
    
    # 处理每个视频的真实标签
    for vid in tqdm.tqdm(os.listdir(gt_path)):                          # 遍历真实标签目录中的每个文件（带进度条）
        file_ptr = open(os.path.join(gt_path, vid), 'r')                # 打开当前视频的真实标签文件
        content = file_ptr.read().split('\n')[:-1]                      # 读取内容并分割（移除最后空行）
        classes = np.zeros([len(content)], dtype=np.int32)              # 创建标签数组
        
        # 将动作名称映射为索引
        for i in range(len(classes)):                                   # 遍历每一帧
            classes[i] = actions_dict[content[i]]                       # 将动作名称转换为索引
        
        # 初始化进度值数组 [动作数, 帧数]
        progress_values = np.zeros([len(actions_dict), len(content)])   # 创建全零进度值数组
        cur_frame = 0  # 当前帧索引初始化
        
        # 计算每个动作片段的进度值
        for k, v in groupby(classes):                                   # 分组连续相同的动作
            segment_length = len(list(v))                               # 计算当前动作片段的长度
            if k not in bg_class:                                       # 如果不是背景类
                cur_progress = (np.arange(segment_length) + 1) / segment_length  # 计算当前片段的进度值（0到1线性增加）
                progress_values[k, cur_frame:cur_frame+segment_length] = cur_progress  # 填充进度值
            cur_frame += segment_length                                 # 更新当前帧索引
        
        # 保存进度值到文件
        np.save(os.path.join(progress_path, vid[:-4]+'.npy'), progress_values)      # 保存为.npy文件（移除.txt扩展名）
    
    print(f"Finished writing progress values for {dataset} in {progress_path}")     # 打印完成信息

if __name__ == '__main__':  # 主程序入口
    # 为gtea数据集生成进度值
    write_progress_values('gtea', [10], ' ')                                        # gtea数据集背景类为10，分隔符为空格
    
    # 为其他数据集生成进度值
    for dataset in ['coffee', 'tea', 'pinwheels', 'oatmeal', 'quesadilla']:         # 遍历数据集列表
        write_progress_values(dataset, [0], '|')                                    # 这些数据集背景类为0，分隔符为竖线

