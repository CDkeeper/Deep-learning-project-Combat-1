import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from model import MyModule
import time

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

train_dataLoader = DataLoader(train_data, batch_size=64)
test_dataLoader = DataLoader(test_data, batch_size=64)


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


myModule = MyModule()
myModule = myModule.to(device)

loss_fun = nn.CrossEntropyLoss()
loss_fun = loss_fun.to(device)  # 由于数据已经部署到gpu，其实损失函数不需要部署

# learning_rate = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(myModule.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 10

# 添加tensorboard
writer = SummaryWriter("./logs_train")
start_time = time.time()
for i in range(epoch):
    print("-----------第{}轮训练开始------------".format(i+1))

    # 训练步骤开始
    for data in train_dataLoader:
        Img, targets = data
        Img = Img.to(device)
        targets = targets.to(device)

        outputs = myModule(Img)
        loss = loss_fun(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{}，Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataLoader:
            Img, targets = data
            Img = Img.to(device)
            targets = targets.to(device)

            outputs = myModule(Img)
            loss = loss_fun(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

        print("整体测试集上的Loss：{}".format(total_test_loss))
        print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
        total_test_step = total_test_step + 1

        torch.save(myModule, "./model_weight/myModule_{}.pth".format(i))
        # torch.save(myModule.state_dict(), "myModule_{}.pth".format(i))
        print("模型已保存")

writer.close()
