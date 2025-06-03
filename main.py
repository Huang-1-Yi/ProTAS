import torch
from model import Trainer  # 从model.py导入Trainer类
from batch_gen import BatchGenerator  # 从batch_gen.py导入BatchGenerator类
import os
import argparse  # 用于解析命令行参数
import random
from eval import evaluate  # 从eval.py导入evaluate函数
import logging  # 日志记录
from datetime import datetime  # 获取当前时间

# 设置设备，优先使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置随机种子以保证结果可复现
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True  # 使卷积操作确定性运行，可能降低性能

# 创建命令行参数解析器
parser = argparse.ArgumentParser()
# 添加各种命令行参数
parser.add_argument('--action', default='train', help="Action to perform: train, predict, predict_online")
parser.add_argument('--dataset', default="ptg", help="Dataset to use")
parser.add_argument('--split', default='1', help="Data split to use")
parser.add_argument('--batch_size', default=1, type=int, help="Batch size for training")
parser.add_argument('--exp_id', default='mstcn', type=str, help="Experiment ID for model saving")
parser.add_argument('--num_epochs', default=50, type=int, help="Number of training epochs")
parser.add_argument('--causal', action='store_true', help="Use causal convolutions")
parser.add_argument('--graph', action='store_true', help="Use graph structures")
parser.add_argument('--learnable_graph', action='store_true', help="Use learnable graph structures")
parser.add_argument('--lr', default=0.0005, type=float, help="Learning rate")
parser.add_argument('--progress_lw', default=1.0, type=float, help="Loss weight for progress prediction")
parser.add_argument('--graph_lw', default=0.1, type=float, help="Loss weight for graph prediction")

# 解析命令行参数
args = parser.parse_args()

# 设置模型参数
num_stages = 4   # 阶段数
num_layers = 10   # 每个阶段的层数
num_f_maps = 64   # 特征图数量
features_dim = 2048  # 输入特征维度
# 从参数中获取批次大小、学习率和训练轮数
bz = args.batch_size
lr = args.lr 
num_epochs = args.num_epochs


# 设置采样率（默认1，即15fps）
# use the full temporal resolution @ 15fps
sample_rate = 1
# 对于50salads数据集，将采样率设为2（因为原始特征为30fps，我们按15fps处理）
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

# 设置文件路径
# 训练和测试的分割文件路径
vid_list_file = f"./data/{args.dataset}/splits/train.split{args.split}.bundle"
vid_list_file_tst = f"./data/{args.dataset}/splits/test.split{args.split}.bundle"
# 特征、标注、进度、图结构、映射文件的路径
features_path = f"./data/{args.dataset}/features/"
gt_path = f"./data/{args.dataset}/groundTruth/"
progress_path = f"./data/{args.dataset}/progress/"
graph_path = f"./data/{args.dataset}/graph/graph.pkl"
mapping_file = f"./data/{args.dataset}/mapping.txt"
# 模型和结果保存目录
model_dir = f"./models/{args.exp_id}/{args.dataset}/split_{args.split}"
results_dir = f"./results/{args.exp_id}/{args.dataset}/epoch{num_epochs}/split_{args.split}"

# 创建模型和结果目录（如果不存在）
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# 设置日志
logger = logging.getLogger('MSTCN')  # 创建名为'MSTCN'的logger
current_time = datetime.now()
log_filename = current_time.strftime("%Y-%m-%d_%H-%M-%S.log")  # 用当前时间生成日志文件
logging.basicConfig(
    level=logging.DEBUG,  # 设置日志级别为DEBUG（记录所有级别）Capture all levels of logging
    filename=os.path.join(model_dir, log_filename),   # 日志文件路径 Name of the log file
    filemode='w',         # 入模式：'w'表示覆盖写入 'w' for overwrite each time; use 'a' to append
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式 Include timestamp, log level, and log message
)
# 添加一个控制台处理器，这样日志也会输出到控制台
ch = logging.StreamHandler()
logger.addHandler(ch)
# 记录命令行参数
logger.info(args)

# 读取动作映射文件（mapping.txt）
with open(mapping_file, 'r') as file_ptr:
    actions = file_ptr.read().split('\n')[:-1]  # 分割每一行，并去掉最后一行空行

# 创建动作字典（动作名称->索引）
actions_dict = dict()
# 根据数据集确定映射文件的分隔符
map_delimiter = '|' if args.dataset in ['coffee', 'tea', 'pinwheels', 'oatmeal', 'quesadilla'] else ' '
# 对于特定数据集，特征需要转置
feature_transpose = True if args.dataset in ['coffee', 'tea', 'pinwheels', 'oatmeal', 'quesadilla'] else False

# 构建动作字典
for a in actions:
    actions_dict[a.split(map_delimiter)[1]] = int(a.split(map_delimiter)[0])# 映射：动作名称 -> 动作ID

# 获取类别数量
num_classes = len(actions_dict)

# 初始化Trainer（训练器）
trainer = Trainer(
    num_stages, num_layers, num_f_maps, features_dim, num_classes, 
    causal=args.causal, logger=logger, progress_lw=args.progress_lw, 
    use_graph=args.graph, graph_lw=args.graph_lw, init_graph_path=graph_path, 
    learnable=args.learnable_graph
)

# Perform the specified action
# 根据命令行参数选择执行的动作
if args.action == "train":
    # 创建批数据生成器
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, progress_path, sample_rate, feature_transpose)
    batch_gen.read_data(vid_list_file)  # 读取训练数据
    # 训练模型
    trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)
    # 训练完成后，使用测试集进行预测
    trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate, feature_transpose, map_delimiter)
    # 评估预测结果
    evaluate(args.dataset, results_dir, args.split, args.exp_id, args.num_epochs)

elif args.action == 'predict':
    # 直接进行预测（测试集）
    trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate, feature_transpose, map_delimiter)
    evaluate(args.dataset, results_dir, args.split, args.exp_id, args.num_epochs)

elif args.action == "predict_online":
    # 在线预测（逐帧处理）
    trainer.predict_online(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate, feature_transpose, map_delimiter)
    evaluate(args.dataset, results_dir, args.split, args.exp_id, args.num_epochs)