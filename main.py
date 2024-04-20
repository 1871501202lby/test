import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torchvision.datasets import CocoDetection
import random

# 定义数据预处理和增强
fixed_size = (224, 224)
data_transforms = transforms.Compose([
    transforms.Resize(fixed_size),    # 调整图像大小为固定大小
    transforms.ToTensor(),            # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

class CustomCocoDetection(CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        super(CustomCocoDetection, self).__init__(root, annFile, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        img, targets = super(CustomCocoDetection, self).__getitem__(index)
        processed_targets = self.process_labels(targets)
        return img, processed_targets

    def process_labels(self, labels):
        processed_labels = []
        for label in labels:
            # 检查标签是否为字典
            if isinstance(label, dict):
                # 获取类别信息
                category_id = label.get('category_id', None)
                if category_id is not None:
                    # 将标签转换为0到1之间的概率值
                    processed_label = torch.tensor(category_id / 90)  # 假设总共有90个类别
                    processed_labels.append(processed_label)
                else:
                    # 如果类别信息缺失，则将其视为一个特殊的类别，如类别编号为0
                    processed_label = torch.tensor(0.0)  # 将缺失的类别信息视为类别编号为0
                    processed_labels.append(processed_label)
            elif isinstance(label, int):
                # 如果标签是整数，则直接转换为张量，并进行归一化处理
                processed_label = torch.tensor(label / 90)  # 假设总共有90个类别
                processed_labels.append(processed_label)
        return processed_labels


# 加载COCO数据集
root = 'D:/MS-COCO/coco2014/val2014/val2014'
annFile = 'D:/MS-COCO/coco2014/annotations/instances_val2014.json'
coco_dataset = CustomCocoDetection(root=root, annFile=annFile, transform=data_transforms)

# 随机采样函数
def random_sample(data, sample_size):
    sample_indices = random.sample(range(len(data)), sample_size)
    return Subset(data, sample_indices)

# 设置采样大小
sample_size = 1000

# 进行随机采样
coco_dataset_sample = random_sample(coco_dataset, sample_size)

def my_collate_fn(batch):
    # 过滤掉不合格的样本
    batch = [(data[0], data[1]) for data in batch if data[1] is not None]
    if len(batch) == 0:
        raise RuntimeError('批次中没有有效的数据')

    # 获取每个样本的图像和标签
    images, labels = zip(*batch)

    # 找到批次中图像的最大尺寸
    max_height = max(img.shape[1] for img in images)
    max_width = max(img.shape[2] for img in images)

    # 将图像调整为相同的最大尺寸，并且进行归一化处理
    images_resized = []
    for idx, img in enumerate(images):
        img_resized = torch.nn.functional.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1]))
        images_resized.append(img_resized)

    # 将图像和标签堆叠成批次
    images = torch.stack(images_resized)

    # 将非空标签堆叠成张量
    processed_labels = [label for label in labels if label]  # 过滤掉空标签
    processed_labels = [label for label in processed_labels if isinstance(label, torch.Tensor)]  # 过滤掉非张量对象
    labels = torch.stack(processed_labels) if processed_labels else None

    return images, labels  # 返回图像张量和标签张量的元组


# 创建 DataLoader 时添加 collate_fn 参数
dataloader = DataLoader(coco_dataset_sample, batch_size=32, collate_fn=my_collate_fn)

# 定义多尺度图像融合模型
class MultiScaleFusionModel(nn.Module):
    def __init__(self):
        super(MultiScaleFusionModel, self).__init__()
        # 使用 ResNet18 作为特征提取器
        self.encoder = nn.Sequential(*list(resnet18(pretrained=True).children())[:-2])
        # 添加额外的卷积层和 Sigmoid 激活函数
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1)  # 使用自适应平均池化层将特征图调整为输出形状为 (batch_size, 1, 1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(x.size(0), -1)  # 将输出展平为(batch_size, 1)



# 训练函数
def train_model(model, dataloader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            if labels is None:
                continue  # 如果标签为 None，则跳过当前样本
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# 主函数
def main():
    # 定义模型、损失函数和优化器
    model = MultiScaleFusionModel()
    criterion = nn.L1Loss()  # 使用平均绝对误差（MAE）损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, dataloader, criterion, optimizer)

if __name__ == "__main__":
    main()
