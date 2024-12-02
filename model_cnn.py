import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Block 1: Conv -> BatchNorm -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 输入通道3，输出通道64
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: Conv -> BatchNorm -> ReLU -> MaxPool
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool
        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)  # CIFAR-10 图像大小经过卷积和池化后为 4x4
        self.dropout = nn.Dropout(0.5)  # 防止过拟合
        self.fc2 = nn.Linear(512, 10)  # 输出10个类别

    def forward(self, x):
        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Block 3
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = self.pool3(F.relu(self.bn3b(self.conv3b(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


model = CNN()
criterion = nn.CrossEntropyLoss()

optimizers = {
    "SGD": optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    "Adam": optim.Adam(model.parameters(), lr=0.001),
    "RMSprop": optim.RMSprop(model.parameters(), lr=0.001)
}

from torch.utils.tensorboard import SummaryWriter

def train_model(optimizer_name, optimizer, epochs=10):
    writer = SummaryWriter(f'runs/cnn/CIFAR10_{optimizer_name}')
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # 每100个批次打印一次
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                writer.add_scalar('Training Loss', running_loss / 100, epoch * len(trainloader) + i)
                running_loss = 0.0
    writer.close()




def evaluate_model():
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    # f1 = f1_score(y_true, y_pred, average='macro')
    return accuracy, precision, recall

if __name__ == '__main__':
    results = {}
    for optimizer_name, optimizer in optimizers.items():
        print(f"Training with {optimizer_name} optimizer...")
        train_model(optimizer_name, optimizer, epochs=10)

        print(f"Evaluating {optimizer_name} optimizer...")
        accuracy, precision, recall = evaluate_model()

        # 存储结果
        results[optimizer_name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            # "F1-Score": f1
        }

    # 打印所有优化器的结果
    print("\nPerformance Metrics for Each Optimizer:")
    for optimizer_name, metrics in results.items():
        print(f"{optimizer_name}: {metrics}")

# 转换结果为 DataFrame
results_df = pd.DataFrame.from_dict(results, orient="index")

# 保存到 CSV 文件
results_df.to_csv("cnn_optimizer_comparison_results.csv", index_label="Optimizer")
print("Results saved to optimizer_comparison_results.csv")
