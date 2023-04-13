import argparse

from Framework import Learner


def get_config():
    parse = argparse.ArgumentParser(description='common supervised learning config')
    parse.add_argument('-task-type-run', type=str, default='test', help='本次训练需要运行的脚本类型')

    # 项目配置参数
    parse.add_argument('-learn-name', type=str, default=None, help='本次训练的名称')
    parse.add_argument('-save-best', type=bool, default=True, help='当得到更好的准确度是否要保存')
    parse.add_argument('-cuda', type=bool, default=True)
    parse.add_argument('-device', type=int, default=2)
    parse.add_argument('-seed', type=int, default=50)

    # 路径参数
    parse.add_argument('-path-token2index', type=str, default='./data/residue2idx.pkl', help='保存字典的位置')
    parse.add_argument('-path-sORFtoken2index', type=str, default='./data/nucleotide2idx.pkl',
                       help='保存核苷酸字典的位置')
    parse.add_argument('-path-train-data', type=str, default='./data/task_data/IL-6/Train.tsv', help='训练数据的位置')
    parse.add_argument('-path-test-data', type=str, default='./data/task_data/IL-6/Validate.tsv', help='测试数据的位置')
    # parse.add_argument('-path-dataset', type=str, default='./data/task_data/Meta Dataset/BPD-ALL-RT',
    #                    help='多分类训练数据的位置')
    parse.add_argument('-path-params', type=str, default=None, help='模型参数路径')
    parse.add_argument('-path-save', type=str, default='./result/', help='保存字典的位置')
    parse.add_argument('-model-save-name', type=str, default='CNN', help='保存模型的命名')
    parse.add_argument('-save-figure-type', type=str, default='png', help='保存图片的文件类型')

    # 数据参数
    parse.add_argument('-num-class', type=int, default=2, help='类别数量')
    # parse.add_argument('-num-class', type=int, default=24, help='类别数量')
    # parse.add_argument('-max-len', type=int, default=52, help='max length of input sequences')
    parse.add_argument('-max-len', type=int, default=209, help='max length of input sequences')
    # parse.add_argument('-dataset', type=str, default='RNA Sequence', help='数据集名称')
    parse.add_argument('-dataset', type=str, default='Peptide Sequence', help='数据集')

    # 框架参数
    parse.add_argument('-mode', type=str, default='train-test', help='训练模式')
    # parse.add_argument('-mode', type=str, default='cross validation', help='训练模式')
    # parse.add_argument('-k-fold', type=int, default=5, help='k折交叉验证')
    parse.add_argument('-interval-log', type=int, default=40, help='经过多少batch记录一次训练状态')
    parse.add_argument('-interval-valid', type=int, default=1, help='经过多少epoch对交叉验证集进行测试')
    parse.add_argument('-interval-test', type=int, default=1, help='经过多少epoch对测试集进行测试')
    # parse.add_argument('-metric', type=str, default='MCC', help='评估指标名称')
    parse.add_argument('-metric', type=str, default='MCC', help='评估指标名称')
    # parse.add_argument('-threshold', type=float, default=0.45, help='指标率阈值')
    parse.add_argument('-threshold', type=float, default=0.40, help='指标率阈值')

    # 训练参数
    parse.add_argument('-model', type=str, default='TextCNN', help='模型名称')
    # parse.add_argument('-model', type=str, default='Transformer Encoder', help='模型名称')
    # parse.add_argument('-optimizer', type=str, default='AdamW', help='优化器名称')
    parse.add_argument('-optimizer', type=str, default='AdamW', help='优化器名称')
    parse.add_argument('-loss-func', type=str, default='contrast loss', help='损失函数名称, CE/contrast loss')
    # parse.add_argument('-loss-func', type=str, default='FL', help='损失函数名称, CE/contrast loss')
    # parse.add_argument('-lr', type=float, default=0.0001, help='学习率')
    # parse.add_argument('-lr', type=float, default=0.0003, help='学习率')
    parse.add_argument('-lr', type=float, default=0.0005, help='学习率')
    parse.add_argument('-reg', type=float, default=0.0025, help='正则化lambda')
    # parse.add_argument('-epoch', type=int, default=50, help='迭代次数')
    parse.add_argument('-epoch', type=int, default=80, help='迭代次数')
    parse.add_argument('-batch-size', type=int, default=8, help='一个batch中有多少个sample')

    # Focal Loss参数
    parse.add_argument('-gamma', type=float, default=2, help='gamma in Focal Loss')
    parse.add_argument('-alpha', type=float, default=0.01, help='alpha in Focal Loss')

    # Transformer Encoder 模型参数
    # # parse.add_argument('-num-layer', type=int, default=3, help='Transformer的Encoder模块的堆叠层数')
    # parse.add_argument('-num-layer', type=int, default=6, help='Transformer的Encoder模块的堆叠层数')
    # parse.add_argument('-dropout', type=float, default=0.5, help='dropout率')
    # parse.add_argument('-static', type=bool, default=False, help='嵌入是否冻结')
    # parse.add_argument('-num-head', type=int, default=8, help='多头注意力机制的头数')
    # parse.add_argument('-dim-embedding', type=int, default=128, help='词（残基）向量的嵌入维度')
    # parse.add_argument('-dim-feedforward', type=int, default=32, help='词（残基）向量的嵌入维度')
    # parse.add_argument('-dim-k', type=int, default=32, help='k/q向量的嵌入维度')
    # parse.add_argument('-dim-v', type=int, default=32, help='v向量的嵌入维度')

    # TextCNN 模型参数
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


def select_fintune_dataset(class_name):
    base_dir = './data/ncPEP'
    path_train_data, path_test_data = None, None
    if class_name == 'Anal_canal_cancer':
        petide_class_name = '/Anal_canal_cancer'
        path_train_data = base_dir + petide_class_name + '/Anal_canal_cancer_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Anal_canal_cancer_test.tsv'
    elif class_name == 'Bile_duct_cancer':
        petide_class_name = '/Bile_duct_cancer'
        path_train_data = base_dir + petide_class_name + '/Bile_duct_cancer_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Bile_duct_cancer_test.tsv'
    elif class_name == 'Bladder_cancer':
        petide_class_name = '/Bladder_cancer'
        path_train_data = base_dir + petide_class_name + '/Bladder_cancer_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Bladder_cancer_test.tsv'
    elif class_name == 'Breast_cancer':
        petide_class_name = '/Breast_cancer'
        path_train_data = base_dir + petide_class_name + '/Breast_cancer_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Breast_cancer_test.tsv'
    elif class_name == 'Colon_cancer':
        petide_class_name = '/Colon_cancer'
        path_train_data = base_dir + petide_class_name + '/Colon_cancer_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Colon_cancer_test.tsv'
    elif class_name == 'Gastric_cancer':
        petide_class_name = '/Gastric_cancer'
        path_train_data = base_dir + petide_class_name + '/Gastric_cancer_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Gastric_cancer_test.tsv'
    elif class_name == 'Kidney_cancer':
        petide_class_name = '/Kidney_cancer'
        path_train_data = base_dir + petide_class_name + '/Kidney_cancer_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Kidney_cancer_test.tsv'
    elif class_name == 'Leukemia':
        petide_class_name = '/Leukemia'
        path_train_data = base_dir + petide_class_name + '/Leukemia_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Leukemia_test.tsv'
    elif class_name == 'Liver_cancer':
        petide_class_name = '/Liver_cancer'
        path_train_data = base_dir + petide_class_name + '/Liver_cancer_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Liver_cancer_test.tsv'
    elif class_name == 'Lung_cancer':
        petide_class_name = '/Lung_cancer'
        path_train_data = base_dir + petide_class_name + '/Lung_cancer_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Lung_cancer_test.tsv'
    elif class_name == 'Ovary_cancer':
        petide_class_name = '/Ovary_cancer'
        path_train_data = base_dir + petide_class_name + '/Ovary_cancer_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Ovary_cancer_test.tsv'
    elif class_name == 'Prostate_cancer':
        petide_class_name = '/Prostate_cancer'
        path_train_data = base_dir + petide_class_name + '/Prostate_cancer_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Prostate_cancer_test.tsv'
    elif class_name == 'Skin_cancer':
        petide_class_name = '/Skin_cancer'
        path_train_data = base_dir + petide_class_name + '/Skin_cancer_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Skin_cancer_test.tsv'
    elif class_name == 'Tongue_cancer':
        petide_class_name = '/Tongue_cancer'
        path_train_data = base_dir + petide_class_name + '/Tongue_cancer_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Tongue_cancer_test.tsv'
    elif class_name == 'Thyroid_cancer':
        petide_class_name = '/Thyroid_cancer'
        path_train_data = base_dir + petide_class_name + '/Thyroid_cancer_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Thyroid_cancer_test.tsv'
    else:
        print('Error, No Such Dataset')
    return path_train_data, path_test_data


def select_sORFs_fintune_dataset(class_name):
    base_dir = './data/sORFs_finetune_dataset'
    path_train_data, path_test_data = None, None
    if class_name == 'Anal_canal_cancer_sORFs':
        sORFs_class_name = '/Anal_canal_cancer_sORFs'
        path_train_data = base_dir + sORFs_class_name + '/Anal_canal_cancer_sORFs_train.tsv'
        path_test_data = base_dir + sORFs_class_name + '/Anal_canal_cancer_sORFs_test.tsv'
    elif class_name == 'Bile_duct_cancer_sORFs':
        petide_class_name = '/Bile_duct_cancer_sORFs'
        path_train_data = base_dir + petide_class_name + '/Bile_duct_cancer_sORFs_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Bile_duct_cancer_sORFs_test.tsv'
    elif class_name == 'Bladder_cancer_sORFs':
        petide_class_name = '/Bladder_cancer_sORFs'
        path_train_data = base_dir + petide_class_name + '/Bladder_cancer_sORFs_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Bladder_cancer_sORFs_test.tsv'
    elif class_name == 'Breast_cancer_sORFs':
        petide_class_name = '/Breast_cancer_sORFs'
        path_train_data = base_dir + petide_class_name + '/Breast_cancer_sORFs_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Breast_cancer_sORFs_test.tsv'
    elif class_name == 'Colon_cancer_sORFs':
        petide_class_name = '/Colon_cancer_sORFs'
        path_train_data = base_dir + petide_class_name + '/Colon_cancer_sORFs_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Colon_cancer_sORFs_test.tsv'
    elif class_name == 'Gastric_cancer_sORFs':
        petide_class_name = '/Gastric_cancer_sORFs'
        path_train_data = base_dir + petide_class_name + '/Gastric_cancer_sORFs_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Gastric_cancer_sORFs_test.tsv'
    elif class_name == 'Kidney_cancer_sORFs':
        petide_class_name = '/Kidney_cancer_sORFs'
        path_train_data = base_dir + petide_class_name + '/Kidney_cancer_sORFs_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Kidney_cancer_sORFs_test.tsv'
    elif class_name == 'Leukemia_sORFs':
        petide_class_name = '/Leukemia_sORFs'
        path_train_data = base_dir + petide_class_name + '/Leukemia_sORFs_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Leukemia_sORFs_test.tsv'
    elif class_name == 'Liver_cancer_sORFs':
        petide_class_name = '/Liver_cancer_sORFs'
        path_train_data = base_dir + petide_class_name + '/Liver_cancer_sORFs_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Liver_cancer_sORFs_test.tsv'
    elif class_name == 'Lung_cancer_sORFs':
        petide_class_name = '/Lung_cancer_sORFs'
        path_train_data = base_dir + petide_class_name + '/Lung_cancer_sORFs_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Lung_cancer_sORFs_test.tsv'
    elif class_name == 'Ovary_cancer_sORFs':
        petide_class_name = '/Ovary_cancer_sORFs'
        path_train_data = base_dir + petide_class_name + '/Ovary_cancer_sORFs_train.tsv'
        path_test_data = base_dir + petide_class_name + '/Ovary_cancer_sORFs_test.tsv'
    elif class_name == 'Prostate_cancer_sORFs':
        sORFs_class_name = '/Prostate_cancer_sORFs'
        path_train_data = base_dir + sORFs_class_name + '/Prostate_cancer_sORFs_train.tsv'
        path_test_data = base_dir + sORFs_class_name + '/Prostate_cancer_sORFs_test.tsv'
    elif class_name == 'Skin_cancer_sORFs':
        sORFs_class_name = '/Skin_cancer_sORFs'
        path_train_data = base_dir + sORFs_class_name + '/Skin_cancer_sORFs_train.tsv'
        path_test_data = base_dir + sORFs_class_name + '/Skin_cancer_sORFs_test.tsv'
    elif class_name == 'Thyroid_cancer_sORFs':
        sORFs_class_name = '/Thyroid_cancer_sORFs'
        path_train_data = base_dir + sORFs_class_name + '/Thyroid_cancer_sORFs_train.tsv'
        path_test_data = base_dir + sORFs_class_name + '/Thyroid_cancer_sORFs_test.tsv'
    elif class_name == 'Tongue_cancer_sORFs':
        sORFs_class_name = '/Tongue_cancer_sORFs'
        path_train_data = base_dir + sORFs_class_name + '/Tongue_cancer_sORFs_train.tsv'
        path_test_data = base_dir + sORFs_class_name + '/Tongue_cancer_sORFs_test.tsv'
    else:
        print('Error, No Such Dataset')
    return path_train_data, path_test_data


config = get_config()
# config.path_params = './result/meta_train_ncPEP_7_3/model/MIMML, Epoch[250.000].pt'
config.data_name = "ACPred-FL"
# path_train_data, path_test_data = select_fintune_dataset(config.data_name)
path_train_data = './data/ncPEP/ACPred-FL/training dataset.tsv'
path_test_data = './data/ncPEP/ACPred-FL/testing dataset.tsv'
config.device = 2
# config.learn_name = 'meta_finetune'
# config.learn_name = 'seek_model_parameter'
config.learn_name = 'test'
# config.learn_name = 'ncPEP_metaTrain_finetune_CL'

config.batch_size = 4
config.test_batch_size = 8
config.model_save_name = config.data_name

config.path_train_data = path_train_data
config.path_test_data = path_test_data
config.output_extend = 'finetune'
config.metric = 'ACC'
config.threshold = 0.48

config.epoch = 200
config.lr = 0.0001

print('=' * 50)
# print('The incoming parameters that take effect are:')
# for key, value in CMD_config.__dict__.items():
#     if value and key in config.__dict__.keys():
#         print('{}: {}'.format(key, value))
#         config.__dict__[key] = value
# print('=' * 50)

for key, value in config.__dict__.items():
    if value is None:
        print('Warning: Value pair is None. [{}]: [{}]'.format(key, value))

# if config.learn_name is None:
#     config.learn_name = 'test_finetune'
# print('Updated Config', config)

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
learner.test_model()
