import argparse

from Framework import Learner


def get_meta_train_config():
    parse = argparse.ArgumentParser(description='common meta learning config')
    parse.add_argument('-task-type-run', type=str, default='test', help='本次训练需要运行的脚本类型')

    # 项目配置参数
    parse.add_argument('-learn-name', type=str, default=None, help='本次训练名称')
    parse.add_argument('-process-name', type=str, default='train (0)', help='Pycharm进程的名称')
    parse.add_argument('-save-best', type=bool, default=True, help='当得到更好的准确度是否要保存')
    parse.add_argument('-cuda', type=bool, default=True)
    parse.add_argument('-device', type=int, default=2)
    parse.add_argument('-seed', type=int, default=50)
    parse.add_argument('-num_workers', type=int, default=4)

    # 路径参数
    parse.add_argument('-path-token2index', type=str, default='./Datasets/residue2idx.pkl', help='保存字典的位置')
    parse.add_argument('-path-meta-dataset', type=str, default='./Datasets/ncPEP_for_meta-learning',
                       help='元学习数据的位置')
    parse.add_argument('-path-params', type=str, default=None, help='模型参数路径')
    parse.add_argument('-path-save', type=str, default='./result/', help='保存字典的位置')
    parse.add_argument('-model-save-name', type=str, default='MIMML', help='保存模型的命名')
    # parse.add_argument('-model-save-name', type=str, default='ProtoNet', help='保存模型的命名')
    parse.add_argument('-save-figure-type', type=str, default='png', help='保存图片的文件类型')

    # 数据参数
    # parse.add_argument('-max-len', type=int, default=207, help='max length of input sequences')
    parse.add_argument('-max-len', type=int, default=52, help='max length of input sequences')
    # parse.add_argument('-max-len', type=int, default=12729, help='max length of input sequences')
    parse.add_argument('-dataset', type=str, default='Peptide Sequence', help='数据集')
    # parse.add_argument('-dataset', type=str, default='RNA Sequence', help='数据集')

    # 框架参数
    parse.add_argument('-mode', type=str, default='meta learning', help='训练模式')
    # parse.add_argument('-valid-start-epoch', type=int, default=0,
    #                    help='meta-train多少个epoch才开始展示meta-valid/meta-test的结果')
    parse.add_argument('-valid-start-epoch', type=int, default=300,
                       help='meta-train多少个epoch才开始展示meta-valid/meta-test的结果')
    parse.add_argument('-valid-interval', type=int, default=5,
                       help='meta-train多少个epoch才进行一次meta-valid/meta-test')
    parse.add_argument('-valid-draw', type=int, default=10, help='meta-train多少个epoch才进行一绘制一次曲线图')
    parse.add_argument('-metric', type=str, default='ACC', help='评估指标名称')
    parse.add_argument('-threshold', type=float, default=0.60, help='准确率阈值')

    # 训练参数
    parse.add_argument('-model', type=str, default='ProtoNet', help='元学习模型名称')
    parse.add_argument('-backbone', type=str, default='TextCNN', help='元学习骨架模型名称')
    parse.add_argument('-optimizer', type=str, default='Adam', help='优化器名称')
    # parse.add_argument('-optimizer', type=str, default='AdamW', help='优化器名称')
    parse.add_argument('-loss-func', type=str, default='FL', help='损失函数名称, CE/FL')

    parse.add_argument('-if-MIM', type=bool, default=True)
    # parse.add_argument('-if-MIM', type=bool, default=False)
    parse.add_argument('-if-transductive', type=bool, default=True, help='inductive or transductive')
    # parse.add_argument('-if-transductive', type=bool, default=False, help='inductive or transductive')
    parse.add_argument('-train-iteration', type=int, default=1, help='meta-train时每个task重复优化多少次')
    # parse.add_argument('-test-iteration', type=int, default=200, help='meta-test测试多少个任务')
    parse.add_argument('-test-iteration', type=int, default=100, help='meta-test测试多少个任务')
    # parse.add_argument('-adapt-iteration', type=int, default=0, help='meta-vald/meta-test时每个task重复优化多少次')
    parse.add_argument('-adapt-iteration', type=int, default=10, help='meta-vald/meta-test时每个task重复优化多少次')
    # parse.add_argument('-adapt-iteration', type=int, default=30, help='meta-vald/meta-test时每个task重复优化多少次')
    parse.add_argument('-valid-iteration', type=int, default=5, help='meta-valid测试多少个任务')
    # parse.add_argument('-valid-iteration', type=int, default=1, help='meta-valid测试多少个任务')

    parse.add_argument('-epoch', type=int, default=251)
    parse.add_argument('-meta-batch-size', type=int, default=10)
    # parse.add_argument('-meta-batch-size', type=int, default=2)
    # parse.add_argument('-lr', type=float, default=0.001)
    # parse.add_argument('-lr', type=float, default=0.0005)
    parse.add_argument('-lr', type=float, default=0.0002)
    # parse.add_argument('-lr', type=float, default=0.0001)
    # parse.add_argument('-adapt-lr', type=float, default=0.001)
    parse.add_argument('-adapt-lr', type=float, default=0.0005)
    # parse.add_argument('-reg', type=float, default=0.001)
    parse.add_argument('-reg', type=float, default=0.0000)

    parse.add_argument('-num-meta-train', type=int, default=24)
    parse.add_argument('-num-meta-valid', type=int, default=10)
    parse.add_argument('-num-meta-test', type=int, default=10)

    # parse.add_argument('-train-way', type=int, default=5)
    # parse.add_argument('-train-shot', type=int, default=5)
    # parse.add_argument('-train-query', type=int, default=15)
    # parse.add_argument('-valid-way', type=int, default=5)
    # parse.add_argument('-valid-shot', type=int, default=5)
    # parse.add_argument('-valid-query', type=int, default=15)
    # parse.add_argument('-test-way', type=int, default=5)
    # parse.add_argument('-test-shot', type=int, default=5)
    # parse.add_argument('-test-query', type=int, default=15)

    parse.add_argument('-train-way', type=int, default=5)
    parse.add_argument('-train-shot', type=int, default=5)
    parse.add_argument('-train-query', type=int, default=10)
    parse.add_argument('-valid-way', type=int, default=2)
    parse.add_argument('-valid-shot', type=int, default=5)
    parse.add_argument('-valid-query', type=int, default=10)
    parse.add_argument('-test-way', type=int, default=0)
    parse.add_argument('-test-shot', type=int, default=0)
    parse.add_argument('-test-query', type=int, default=0)

    # 损失系数
    parse.add_argument('-alpha', type=float, default=0.1)
    # parse.add_argument('-alpha', type=float, default=0.05)
    parse.add_argument('-lamb', type=float, default=0.1)
    # parse.add_argument('-lamb', type=float, default=0.05)
    parse.add_argument('-temp', type=float, default=20)
    # parse.add_argument('-temp', type=float, default=15)

    # 优化器系数
    # parse.add_argument('-if-lr-scheduler', type=bool, default=False)
    # parse.add_argument('-lr-step-size', type=int, default=10)
    # parse.add_argument('-gamma', type=int, default=0.9)

    # TextCNN Finetune 模型参数
    parse.add_argument('-dim-embedding', type=int, default=128, help='词（残基）向量的嵌入维度')
    parse.add_argument('-dropout', type=float, default=0.5, help='dropout率')
    parse.add_argument('-static', type=bool, default=False, help='嵌入是否冻结')
    parse.add_argument('-num-filter', type=int, default=128, help='卷积核的数量')
    parse.add_argument('-filter-sizes', type=str, default='1,2,4,8,16,24,32', help='卷积核的尺寸')
    parse.add_argument('-dim-cnn-out', type=int, default=128, help='CNN模型的输出维度')

    # parse.add_argument('-output-extend', type=str, default='pretrain', help='CNN后是否再接一层')
    # parse.add_argument('-output-extend', type=str, default='finetune', help='CNN后是否再接一层')

    config = parse.parse_args()
    return config


config = get_meta_train_config()
config.learn_name = ''

# config.path_params = './result/pretrain/model/CNN, Epoch[20.000].pt'
# config.threshold = 0.60

print('=' * 50)
# print('The incoming parameters that take effect are:')
# for key, value in CMD_config.__dict__.items():
#     if value and key in config.__dict__.keys():
#         print('{}: {}'.       format(key, value))
#         config.__dict__[key] = value
# print('=' * 50)

for key, value in config.__dict__.items():
    if value is None:
        print('Warning: Value pair is None. [{}]: [{}]'.format(key, value))

print('Updated Config', config)

learner = Learner.Learner(config)
learner.setIO()
learner.setVisualization()
learner.load_data()
learner.init_model()
learner.load_params()
learner.adjust_model()
learner.init_optimizer()
learner.def_loss_func()
learner.train_model()
# learner.test_model()
