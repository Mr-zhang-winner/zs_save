import torch

from model import RNN, DAtt, CSEN, T_UNet, DAtt_ZS
from utils.BaseClass import ModelTrainer
from utils.TrainUtils import init_weights

random_seed = torch.initial_seed()
print("随机种子：" + str(random_seed))


class RNNTrainer(ModelTrainer):
    def __init__(self, model_name="newrnn", epochs=300):
        super(RNNTrainer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RNN().to(self.device)  # 创建RNN模型实例并移动到GPU
        # self.model = RNN()  # 创建RNN模型实例
        init_weights(self.model, init_type="kaiming")

        # 模型初始化
        self.bce, self.cross_entropy, self.optimizer = None, None, None
        self.initialize_model()  # 初始化模型
        # 定义训练需要的参数
        self.epochs = epochs
        self.best_f1 = -float("inf")
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.precision, self.recall, self.f1 = 0.0, 0.0, 0.0
        self.epoch = 0
        self.th = 0.4

        self.save_path = f"./new{model_name}_{epochs}.pth"


class DATTrainer(ModelTrainer):
    def __init__(self, model_name="newDAtt"):
        super(DATTrainer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DAtt().to(self.device)  # 创建DAtt()模型实例并移动到GPU
        # self.model = DAtt()  # 创建RNN模型实例
        init_weights(self.model, init_type="kaiming")
        self.save_path = f"./new{model_name}.pth"
        # 模型初始化
        self.bce, self.cross_entropy, self.optimizer = None, None, None
        self.initialize_model()  # 初始化模型
        # 定义训练需要的参数
        self.epochs = 300
        self.best_f1 = -float("inf")
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.precision, self.recall, self.f1 = 0.0, 0.0, 0.0
        self.epoch = 0
        self.th = 0.4


class CSENTrainer(ModelTrainer):
    def __init__(self, model_name="newcsen-100", epochs=300):
        super(CSENTrainer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CSEN().to(self.device)  # 创建CSEN()模型实例并移动到GPU
        # self.model = CSEN()
        init_weights(self.model, init_type="kaiming")

        self.save_path = f"./new{model_name}.pth"
        # 模型初始化
        self.bce, self.cross_entropy, self.optimizer = None, None, None
        self.initialize_model()  # 初始化模型
        # 定义训练需要的参数
        self.epochs = epochs
        self.best_f1 = -float("inf")
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.precision, self.recall, self.f1 = 0.0, 0.0, 0.0
        self.epoch = 0
        self.th = 0.4


class T_UNetTrainer(ModelTrainer):
    def __init__(self, model_name="newT_UNet", epochs=300):
        super(T_UNetTrainer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T_UNet().to(self.device)  # 创建T_UNet()模型实例并移动到GPU
        init_weights(self.model, init_type="kaiming")

        # 模型初始化
        self.bce, self.cross_entropy, self.optimizer = None, None, None
        self.initialize_model()  # 初始化模型
        # 定义训练需要的参数
        self.epochs = epochs
        self.best_f1 = -float("inf")
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.precision, self.recall, self.f1 = 0.0, 0.0, 0.0
        self.epoch = 0
        self.th = 0.4
        self.save_path = f"./new{model_name}_{self.epochs}.pth"


class DATTTrainer(ModelTrainer):
    def __init__(self, model_name="newT_UNet", epochs=300):
        super(DATTTrainer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T_UNet(1).to(self.device)  # 创建CSEN()模型实例并移动到GPU
        init_weights(self.model, init_type="kaiming")

        # 模型初始化
        self.bce, self.cross_entropy, self.optimizer = None, None, None
        self.initialize_model()  # 初始化模型
        # 定义训练需要的参数
        self.epochs = epochs
        self.best_f1 = -float("inf")
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.precision, self.recall, self.f1 = 0.0, 0.0, 0.0
        self.epoch = 0
        self.th = 0.4
        self.save_path = f"./new{model_name}_{self.epochs}.pth"


class DAtt_ZSrainer(ModelTrainer):
    def __init__(self, model_name="newDAtt_ZS", epochs=300):
        super(DAtt_ZSrainer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DAtt_ZS().to(self.device)  # 创建CSEN()模型实例并移动到GPU
        init_weights(self.model, init_type="kaiming")

        # 模型初始化
        self.bce, self.cross_entropy, self.optimizer = None, None, None
        self.initialize_model()  # 初始化模型
        # 定义训练需要的参数
        self.epochs = epochs
        self.best_f1 = -float("inf")
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.precision, self.recall, self.f1 = 0.0, 0.0, 0.0
        self.epoch = 0
        self.th = 0.4
        self.save_path = f"./new{model_name}_{self.epochs}.pth"


if __name__ == "__main__":
    epochs = 300
    trainer_1 = RNNTrainer(epochs=epochs)
    trainer_2 = CSENTrainer(epochs=epochs)
    trainer_3 = T_UNetTrainer(epochs=epochs)
    trainer_4 = DAtt_ZSrainer(epochs=epochs)

    for epoch in range(epochs):
        # trainer_1.epoch = epoch
        trainer_2.epoch = epoch
        trainer_3.epoch = epoch
        trainer_4.epoch = epoch

        print("DAtt_ZS_trainer_cls：")
        trainer_4.train_model()
        # print("T_UNet_trainer_cls：")
        # trainer_3.train_model()
        # trainer_1.train_model_with_pseudo_label(trainer_3)
        # trainer_3.train_model_with_pseudo_label(trainer_1)


        # # 前2个epoch使用类别标签训练DATT_trainer_1_cls和CSEN_trainer_2
        # if epoch < 10:
        #     print("DAtt_trainer_4_cls：")
        #     # trainer_2.train_model_with_pseudo_label(trainer_3)
        #     # trainer_3.train_model_with_pseudo_label(trainer_1)
        #     # trainer_4.train_model_with_pseudo_label(trainer_3)
        #     trainer_4.train_model()
        #     # print("CSEN_trainer_2_cls：")
        #     # trainer_2.train_model()
        #     print("T_UNet_trainer_3_cls：")
        #     trainer_3.train_model()
        #
        # else:
        #     # 后面加入伪标签训练 5个epoch
        #     if (epoch - 10) % 10 < 5:
        #         print("DAtt_trainer_4：")
        #         trainer_4.train_model_with_pseudo_label(trainer_3)
        #
        #     else:
        #         # print("CSEN_trainer_2：")
        #         # trainer_2.train_model_with_pseudo_label(trainer_1)
        #         print("T_UNet_trainer_3：")
        #         trainer_3.train_model_with_pseudo_label(trainer_4)
