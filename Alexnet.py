import os
import sys
import json
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        # 定义特征提取层，由多个卷积层和池化层组成
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        # 定义分类器层，由多个全连接层和Dropout层组成
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        # 如果init_weights为True，则初始化权重
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # 初始化权重的私有方法
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 对卷积层的权重使用Kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 对卷积层的偏置使用常数初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 对全连接层的权重使用正态分布初始化
                nn.init.normal_(m.weight, 0, 0.01)
                # 对全连接层的偏置使用常数初始化
                nn.init.constant_(m.bias, 0)

def main():
    #采用gpu还是cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    print(os.path.join(os.getcwd())) #D:\code\MatrixProject
    #图像分类任务中常见的数据增强和预处理步骤。
    # 定义一个名为data_transform的字典，用于存储训练集和验证集的数据转换操作
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     # 随机水平翻转图像
                                     transforms.RandomHorizontalFlip(),
                                     # 将PIL图像或者numpy.ndarray转换为Tensor
                                     transforms.ToTensor(),
                                     # 标准化Tensor，这里使用的均值和标准差都是(0.5, 0.5, 0.5)
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    # 使用os模块中的函数获取数据的根目录路径
    data_root = os.path.abspath(os.path.join(os.getcwd(),""))  # get data root path
    print(data_root)
    # 创建或指定 CIFAR-10 数据集保存的路径
    image_path = os.path.join(data_root, "cifar10_data")
    os.makedirs(image_path, exist_ok=True)  # 如果目录不存在，则创建
    # assert 断言为真就保存图片路径，为假就抛出异常
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    #使用 datasets.ImageFolder 类来加载训练集的图像数据。这个类可以自动地从文件夹中加载图像，并且可以根据文件夹的名称来自动地为图像分配标签。
    train_dataset = datasets.CIFAR10(root=image_path, train=True,
                                     transform=data_transform["train"], download=True)
    validate_dataset = datasets.CIFAR10(root=image_path, train=False,
                                        transform=data_transform["val"], download=True)
    #训练集的图片数量
    train_num = len(train_dataset)
    val_num = len(validate_dataset)

    # CIFAR-10 类别名称及其索引
    cifar10_classes = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }

    # 将字典转换为 JSON 格式的字符串，格式化输出（缩进 4 格）
    json_str = json.dumps(cifar10_classes, indent=4)
    # 将类索引字典写入 'class_indices.json' 文件
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    print("CIFAR-10 类别索引已保存到 class_indices.json 文件中。")

    #每次迭代处理32张样本数量
    batch_size = 32
    #数据加载器的工作进程数
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    #创建两个数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,val_num))

    #创建一个 AlexNet 模型实例
    net = AlexNet(num_classes=10, init_weights=True)
    net.to(device)
    #使用 nn.CrossEntropyLoss() 定义交叉熵损失函数,它结合了softmax操作和负对数似然损失。
    loss_function = nn.CrossEntropyLoss()
    #创建一个Adam优化器，设置学习率为 0.0002
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 10
    # 设置保存训练好的模型的路径
    save_path = './AlexNet.pth'
    # 初始化最佳准确率变量，用于记录验证阶段的最高准确率
    best_acc = 0.0
    # 计算训练步骤的数量，等于训练数据加载器中的批次总数
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # 设置模型为训练模式
        net.train()
        # 初始化本轮训练的累计损失
        running_loss = 0.0
        # 使用tqdm创建一个进度条，用于显示训练进度
        train_bar = tqdm(train_loader, file=sys.stdout)
        # 遍历训练数据加载器中的每个批次
        for step, data in enumerate(train_bar):
            # 从数据中提取图像和标签
            images, labels = data
            # 清除之前的梯度信息
            optimizer.zero_grad()
            # 将图像数据移动到设备上，并进行前向传播，获取模型输出
            outputs = net(images.to(device))
            # 计算损失函数
            loss = loss_function(outputs, labels.to(device))
            # 反向传播，计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()
            # 累计本轮的损失，用于后续计算平均损失
            running_loss += loss.item()
            # 更新进度条的描述，显示当前epoch、总epochs和当前批次的损失
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # 将模型设置为评估模式，这会禁用dropout和batch normalization层的训练行为
        net.eval()
        # 初始化准确率计数器
        acc = 0.0  # accumulate accurate number / epoch
        # 使用torch.no_grad()禁用梯度计算，因为在评估阶段不需要计算梯度
        with torch.no_grad():
            # 创建验证数据的进度条
            val_bar = tqdm(validate_loader, file=sys.stdout)
            # 遍历验证数据加载器中的每个批次
            for val_data in val_bar:
                val_images, val_labels = val_data
                # 将图像数据移动到设备上，并进行前向传播，获取模型输出
                outputs = net(val_images.to(device))
                # 获取预测结果，即输出中最大值的索引
                predict_y = torch.max(outputs, dim=1)[1]
                # 计算预测正确的数量，并累加到acc变量中
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        # 计算整个验证集的平均准确率
        val_accurate = acc / val_num
        # 打印当前epoch的训练损失和验证准确率
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        # 如果当前epoch的验证准确率高于之前记录的最佳准确率，则更新最佳准确率，并保存模型
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
