import numpy as np                  # 数值计算库
import os                           # 操作系统接口
from itertools import groupby       # 迭代器分组工具
import tqdm                         # 进度条显示
import pickle                       # 对象序列化

def write_graph_from_transcripts(dataset, bg_class=[0], map_delimiter=' '):
    """
    该函数从指定数据集的标注文件中读取动作序列，生成前继和后继关系矩阵，并保存为图文件。
    Generate and write a task graph from transcripts of actions.

    Parameters:
    - dataset (str): 数据集名称(如 'gtea')The name of the dataset.
    - bg_class (list): 背景动作标签(默认[0])List of background class labels to ignore.
    - map_delimiter (str): 映射文件分隔符(默认空格)Delimiter used in the mapping file.

    关键概念说明
        ​前继矩阵 (pre_mat)​​
            pre_mat[i][j] 表示动作i后接动作j的条件概率
            计算方式:出现i→j次数 / 动作j总出现次数
        ​后继矩阵 (suc_mat)​​
            suc_mat[j][i] 表示动作j前有动作i的条件概率
            本质上是pre_mat的转置视角
        ​动作序列处理​
            背景过滤:移除无效动作(如'背景'类)
            重复合并:连续相同动作合并为单个实例
            序列示例:原序列 [A,A,B,C,C] → 处理后 [A,B,C]
        ​应用场景​
            动作预测:基于当前动作预测下一个可能动作
            异常检测:识别不符合概率关系的异常动作序列
            流程优化:分析任务动作的依赖关系
        此代码生成的关系图可用于时间动作定位、动作预测等任务，提供动作间的概率依赖关系。
    """
    gt_path = os.path.join('data', dataset, 'groundTruth')          # 标注文件路径
    mapping_file = os.path.join("data", dataset, "mapping.txt")     # 动作映射文件路径
    graph_path = os.path.join('data', dataset, 'graph')             # 图保存路径
    os.makedirs(graph_path, exist_ok=True)                          # 创建保存目录(存在则不报错)
    
    # Create a dictionary to map actions to indices
    actions_dict = dict()                                           # 初始化动作名称到ID的映射字典
    with open(mapping_file, 'r') as f:
        for line in f:
            actions = line.strip().split(map_delimiter)             # 分割每行
            actions_dict[actions[1]] = int(actions[0])              # 格式:{动作名: ID}
    
    # Initialize matrices for predecessor and successor relationships
    pre_mat = np.zeros([len(actions_dict), len(actions_dict)])      # 前继关系矩阵(A->B)
    suc_mat = np.zeros([len(actions_dict), len(actions_dict)])      # 后继关系矩阵(B->A)
    count = np.zeros([len(actions_dict)])                           # 各动作出现总计数器
    
    # 加工每个视频的标注文件
    # Process each video in the ground truth path 
    for vid in tqdm.tqdm(os.listdir(gt_path)):                      # 遍历所有标注文件(带进度条)
        file_ptr = open(os.path.join(gt_path, vid), 'r')
        content = file_ptr.read().split('\n')[:-1]                  # 读取文件内容(去末尾空行)
        classes = np.zeros([len(content)], dtype=np.int32)          # 初始化动作ID数组
        
        # 转换动作名称为ID
        for i in range(len(classes)):
            classes[i] = actions_dict[content[i]]
        
        # 过滤背景动作并合并连续重复动作
        classes_wo_bg = [a for a in classes if a not in bg_class]   # 移除背景动作
        transcript = [k for k, v in groupby(classes_wo_bg)]         # 合并连续重复动作
        
        # 更新计数器和关系矩阵
        for a in transcript:
            count[a] += 1                                           # 更新动作计数
        # 遍历动作序列中的连续动作对
        for pre_action, suc_action in zip(transcript[:-1], transcript[1:]):
            pre_mat[pre_action, suc_action] += 1                    # 前继关系计数 (A->B)
            suc_mat[suc_action, pre_action] += 1                    # 后继关系计数 (B->A)
    
    # Normalize the matrices
    # after normalization, pre_mat and suc_mat are not symmetric
    pre_mat = pre_mat / np.maximum(count[None, :], 1e-5)            # 按后继动作出现次数归一化
    suc_mat = suc_mat / np.maximum(count[None, :], 1e-5)            # 同上(使用None扩展维度)
    # 解释:pre_mat[i,j] = P(动作j|动作i) 概率表示
    
    # Save the graph
    graph = {'matrix_pre': pre_mat, 'matrix_suc': suc_mat}          # 构建结果字典
    with open(os.path.join(graph_path, 'graph.pkl'), 'wb') as f:
        pickle.dump(graph, f)                                       # 序列化保存到文件
    print(f"Finished writing graph for {dataset} in {graph_path}")

if __name__ == '__main__':
    # 处理不同数据集(指定不同背景标签和分隔符)
    write_graph_from_transcripts('gtea', [10], ' ')                 # gtea数据集背景类=10
    for dataset in ['coffee', 'tea', 'pinwheels', 'oatmeal', 'quesadilla']:
        write_graph_from_transcripts(dataset, [0], '|')             # 其他数据集背景类=0，分隔符为|

