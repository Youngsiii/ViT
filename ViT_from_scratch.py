import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time
import random
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm


RANDOM_SEED = 42  # 设置随机种子，确保实验的可重复性
IMG_SIZE = 28     # 假设image是正方形的，长宽均为28
PATCH_SIZE = 4    # 假设patch也是正方形的，长宽均为4
IN_CHANNELS = 1   # MNIST图像通道数可为1
DROPOUT = 0.001   # 将多少比例的参数设置为0
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # (28 // 4) ** 2 = 49 总共有49个patches/tokens
EMBED_DIM = (PATCH_SIZE ** 2) * IN_CHANNELS  # 16  # 每个patch/token的嵌入维度设为16
NUM_HEADS = 8
HIDDEN_DIM = 768
ACTIVATION = "gelu"
NUM_CLASSES = 10
NUM_ENCODERS = 4
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.9, 0.999)
ADAM_WEIGHT_DECAY = 0
EPOCHS = 150

# 让实验具有可重复性
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = "cuda" if torch.cuda.is_available() else "cpu"

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super(PatchEmbedding, self).__init__()
        # embed_dim: 每个token的嵌入维度
        # patch_size: 划分的每个patch的大小(ph, pw)
        # num_patches: 经过划分后patches的总数
        # dropout: 需要进行dropout的比例
        # in_channels: 输入图片的通道数，使用卷积将其转换为embed_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.in_channels = in_channels

        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=embed_dim,
                      kernel_size=patch_size,
                      stride=patch_size),
            nn.Flatten(start_dim=2),
        )
        self.cls_token = nn.Parameter(data=torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
        self.position_embedding = nn.Parameter(data=torch.randn(size=(1, 1+num_patches, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):   # x:(BS, 1, H, W)
        x = self.patcher(x).permute(0, 2, 1)   # (BS, 1, H, W) -> (BS, embed_dim, H//ph, W//pw) -> (BS, embed_dim, H//ph * W//pw=num_patches) -> (BS, num_patches, embed_dim)
        cls_token = self.cls_token.expand(size=(x.shape[0], -1, -1))   # (1, 1, embed_dim) -> (BS, 1, embed_dim)
        x = torch.cat([cls_token, x], dim=1)   # (BS, 1+num_patches, embed_dim)
        x = x + self.position_embedding      # (BS, 1+num_patches, embed_dim) + (1, 1+num_patches, embed_dim)广播 -> (BS, 1+num_patches, embed_dim)
        x = self.dropout(x)                  # (BS, 1+num_patches, embed_dim)
        return x                             # (BS, 1+num_patches, embed_dim)


patchembedding = PatchEmbedding(embed_dim=EMBED_DIM,
                                patch_size=PATCH_SIZE,
                                num_patches=NUM_PATCHES,
                                dropout=DROPOUT,
                                in_channels=IN_CHANNELS).to(device)

x = torch.randn(512, 1, 28, 28).to(device)    # 随机生成512张1通道28×28的图片
print(patchembedding(x).shape)                # 应该是(512, 1+49, 16)





class ViT(nn.Module):
    def __init__(self,
                 embed_dim,
                 patch_size,
                 num_patches,
                 dropout,
                 in_channels,
                 num_heads,
                 hidden_dim,
                 activation,
                 num_encoder_layers,
                 num_classes):
        """
        embed_dim: 每个patch/token的嵌入维度
        patch_size: 每个patch的大小 (ph, pw)
        num_patches: 划分的patches总数 H//ph * W//pw
        dropout: dropout的比例
        in_channels: 输入图片的通道数
        num_heads: 多头注意力的头数
        hidden_dim: Encoder中FFN的隐藏层维度
        activation: 激活函数
        num_encoder_layers: 编码器的个数
        num_classes: 需要分类的类别数
        """
        super(ViT, self).__init__()
        self.embeddings_block = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)

        # 构造Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, activation=activation, batch_first=True, norm_first=True)
        # 堆叠Transformer编码器层构造Transformer编码器
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_encoder_layers)

        # 构造MLP分类头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes),
        )


    def forward(self, x):  # (BS, 1, H, W)
        x = self.embeddings_block(x)   # (BS, 1, H, W) -> (BS, 1+num_patches, embed_dim)
        x = self.encoder_blocks(x)   # (BS, 1+num_patches, embed_dim) -> (BS, 1+num_patches, embed_dim)  TransformerEncoder不改变x的形状
        x = self.mlp_head(x[:, 0, :])  # x[:, 0, :]  (BS, embed_dim) -> (BS, num_classes)
        return x     # (BS, num_classes)



model = ViT(EMBED_DIM, PATCH_SIZE, NUM_PATCHES, DROPOUT, IN_CHANNELS, NUM_HEADS, HIDDEN_DIM, ACTIVATION, NUM_ENCODERS, NUM_CLASSES).to(device)
x = torch.randn((512, 1, 28, 28)).to(device)    # 随机生成512张1通道28×28的图片
print(model(x).shape)                           # 应该是(512, 10)





# 数据处理部分
# 此MNIST数据集由csv文件构成，train.csv中每一行代表一张图片信息
train_df = pd.read_csv("digit-recognizer/train.csv")
test_df = pd.read_csv("digit-recognizer/test.csv")
sample_submission_df = pd.read_csv("digit-recognizer/sample_submission.csv")

print(train_df.head())
print(test_df.head())
print(sample_submission_df.head())

# 从训练数据集train_df中划分出训练集和验证集
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=RANDOM_SEED, shuffle=True)


class MNISTTrainDataset(Dataset):
    def __init__(self, images, labels, indices):
        self.images = images        # Numpy数组(num_image, 784)
        self.labels = labels        # Numpy数组(num_image)
        self.indices = indices      # Numpy数组(num_image)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),               # transforms只能针对PIL图像或Tensor图像进行变换，如果要对numpy数组图像进行变换要先将其转换为PIL图像
            transforms.RandomRotation(15),         # PIL图像(28, 28)
            transforms.ToTensor(),                 # Tensor(1, 28, 28)  # 增加的1应该是通道数
            transforms.Normalize([0.5], [0.5]),    # Tensor(1, 28, 28)
        ])


    def __len__(self):
        return len(self.images)    # return num_image

    def __getitem__(self, idx):
        image = self.images[idx].reshape((28, 28)).astype(np.uint8)    # Numpy数组(784)->uint8(28, 28)
        label = self.labels[idx]
        index = self.indices[idx]
        image = self.transform(image)   # uint8(28, 28) -> PIL(28, 28) -> tensor(1, 28, 28)
        return {"image": image, "label": label, "indice": index}



class MNISTValDataset(Dataset):
    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape((28, 28)).astype(np.uint8)
        label = self.labels[idx]
        index = self.indices[idx]
        image = self.transform(image)

        return {"image": image, "label":label, "index": index}


class MNISTTestDataset(Dataset):
    def __init__(self, images, indices):
        self.images = images                       # Numpy数组(num_image, 784)
        self.indices = indices                     # Numpy数组(num_image)
        self.transform = transforms.Compose([
            transforms.ToTensor(),                 # numpy(28, 28)->tensor(1, 28, 28)
            transforms.Normalize([0.5], [0.5]),    # tensor(1, 28, 28)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape((28, 28)).astype(np.uint8)     # Numpy数组(784)->uint8(28, 28)
        index = self.indices[idx]
        image = self.transform(image)
        return {"image": image, "index": index}    # image: tensor(1, 28, 28), index: Numpy数组(num_image)



# 分别画出训练集、验证集和测试集的一张图片
fig, axarr = plt.subplots(1, 3)    # 1行3列子图构成的画布， axarr为三个子图坐标轴数组axarr = [axarr[0], axarr[1], axarr[2]]
train_dataset = MNISTTrainDataset(
    images=train_df.iloc[:, 1:].values.astype(np.uint8),   # Numpy数组(num_image, 784)
    labels=train_df.iloc[:, 0].values,                     # Numpy数组(num_image)
    indices=train_df.index.values                          # Numpy数组(num_image)
)

axarr[0].imshow(train_dataset[0]["image"].squeeze(), cmap="gray")     # tensor(1, 28, 28)->tensor(28, 28)  gray image
axarr[0].set_title("train image")
print("-"*30)

val_dataset = MNISTValDataset(
    images=val_df.iloc[:, 1:].values.astype(np.uint8),
    labels=val_df.iloc[:, 0].values,
    indices=val_df.index.values
)

axarr[1].imshow(val_dataset[0]["image"].squeeze(), cmap="gray")
axarr[1].set_title("val image")
print("-"*30)

test_dataset = MNISTTestDataset(
    images=test_df.values.astype(np.uint8),
    indices=test_df.index.values,
)
axarr[2].imshow(test_dataset[0]["image"].squeeze(), cmap="gray")
axarr[2].set_title("test image")
print("-"*30)

plt.show()





# 开始训练模型
# 构造数据加载器
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 构造损失函数和优化器
criterion = nn.CrossEntropyLoss()   # 采用交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=ADAM_WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)
# 经过5个epoch如果loss不下降，learning_rate就乘以0.1
# verbose=True每当学习率被减少时，调度器都会打印一条消息通知用户学习率已经被调整



start = time.time()
for epoch in tqdm(range(EPOCHS), position=0, leave=True):
    model.train()                      # 将model设置为训练模式用于训练
    train_running_loss = 0             # 用于保存每个epoch训练的平均每个batch的损失
    train_labels = []                  # 用于保存训练数据集中所有图片的真实label
    train_preds = []                   # 用于保存训练数据集中所有图片的预测label
    for batch_idx, image_label in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        # image_label:{"image":tensor(BS, 1, 28, 28), "label":tensor(BS), "index": tensor(BS)}
        images = image_label["image"].to(device)        # (BS, 1, 28, 28)
        labels = image_label["label"].to(device)        # (BS)
        preds = model(images)                           # (BS, num_classes)
        preds_label = torch.argmax(preds, dim=-1)       # (BS)

        # 将所有真实和预测label进行保存  .extend()用于向列表中追加一个列表的所有元素
        train_labels.extend(labels.detach().cpu())      # list(BS)     [tensor(5), tensor(3), ..., tensor(6)]
        train_preds.extend(preds_label.detach().cpu())  # list(BS)     [tensor(5), tensor(3), ..., tensor(8)]

        # 前向传播
        loss = criterion(preds, labels)
        train_running_loss += loss.item()               # train_running_loss保存这个epoch中所有图片预测的loss，每次加上一个batch的图片预测loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / (batch_idx + 1)   # 表示训练完这个epoch的所有batch的图片的平均每个batch的预测loss


    # 在验证集上进行验证
    model.eval()                # 将模型设置为验证模式
    val_running_loss = 0
    val_labels = []
    val_preds = []
    with torch.no_grad():       # 设置上下文管理器禁止执行梯度计算，告诉 PyTorch 不要追踪计算图中的梯度信息，主要用于推理阶段或计算指标时不想记录任何计算操作的历史信息，从而节省内存
        for batch_idx, image_label in enumerate(tqdm(val_dataloader, position=0, leave=True)):
            # image_label={"image": tensor(BS, 1, 28, 28), "label": tensor(BS), "index": tensor(BS)}
            images = image_label["image"].to(device)      # (BS, 1, 28, 28)
            labels = image_label["label"].to(device)      # (BS)
            preds = model(images)                         # (BS, num_classes)
            preds_label = torch.argmax(preds, dim=-1)     # (BS)
            val_labels.extend(labels.detach().cpu())          # list(BS)
            val_preds.extend(preds_label.detach().cpu())      # list(BS)

            loss = criterion(preds, labels)
            val_running_loss += loss.item()

        val_loss = val_running_loss / (batch_idx + 1)

    scheduler.step(val_loss)

    print("-" * 30)
    print(f"Train Loss:    epoch:{epoch}, train_loss:{train_loss:.4f}")
    print(f"Val Loss:      epoch:{epoch}, val_loss:{val_loss:.4f}")
    # train_preds和train_labels是两个列表
    # [tensor(5), tensor(3), ..., tensor(6)]
    # [tensor(5), tensor(3), ..., tensor(8)]
    train_correct = 0
    for x, y in zip(train_preds, train_labels):
        if x == y:
            train_correct += 1

    train_accuracy = train_correct / len(train_labels)

    # val_preds和val_labels也是两个列表
    val_correct = 0
    for x, y in zip(val_preds, val_labels):
        if x == y:
            val_correct += 1

    val_accuracy = val_correct / len(val_labels)

    print(f"Train Accuracy:   epoch:{epoch}, train_accuracy:{train_accuracy:.4f}")
    print(f"Val Accuracy:     epoch:{epoch}, val_accuracy:{val_accuracy:.4f}")
    if val_accuracy > 0.98:
        state_dict = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
        torch.save(state_dict, f"vit_{val_accuracy: .2f}_model.pth")
        break




end = time.time()
print(f"Train Time: {end-start:.2f}s")




torch.cuda.empty_cache()
# 释放未被使用的缓存，重新分配。一定程度上解决了batch_size增大时显存爆炸的问题，满足暂时需求
# 非必要情况不要经常用，比如可以在训练结束时调用一下


ids = []       # 记录测试图片的ImageId
labels = []    # 记录测试图片的预测Label
imgs = []      # 记录测试图片
model.eval()   # 将模型设置为验证模式用来对测试图片进行测试
with torch.no_grad():
    for batch_idx, sample in enumerate(tqdm(test_dataloader, position=0, leave=True)):
        # sample={"image": tensor(BS, 1, 28, 28), "index": tensor(BS)}
        images = sample["image"].to(device)    # (BS, 1, 28, 28)
        indexes = sample["index"]              # (BS)
        ids.extend([int(i)+1 for i in indexes])     # index是从0开始的，但是ImageId要从1开始，所以加1  ids=[1, 2, 3, ...]
        outputs = model(images)                # (BS, num_classes)
        preds_label = torch.argmax(outputs, dim=-1)    # (BS)
        labels.extend([int(i) for i in preds_label])   # labels=[5, 2, 3, 8, ...]
        imgs.extend(images.detach().cpu())       # 保存图片

# 画出测试数据集中前6张图片和对应的预测label
fig, axarr = plt.subplots(2, 3)
counter = 0   # 显示的图片的id
for i in range(2):
    for j in range(3):
        axarr[i][j].imshow(imgs[counter].squeeze(), cmap="gray")    # (1, 28, 28)->(28, 28)  "gray"
        axarr[i][j].set_title(f"predicted {labels[counter]}")
        counter += 1

# 显示所有图片
plt.show()


# 将idx和labels写入到submission.csv中
sample_submission_df = pd.DataFrame(list(zip(ids, labels)), columns=["ImageId", "Labels"])
sample_submission_df.to_csv("submission.csv", index=False)   # 不用显示index
print(sample_submission_df.head())

















