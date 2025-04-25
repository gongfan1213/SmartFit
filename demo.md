# Project 2: SmartRecipe

Overview

SmartRecipe is a revolutionary app that simplifies cooking by using advanced image recognition to identify ingredients from user photos. More than just a recipe finder, it suggests personalized dishes based on available ingredients and provides detailed nutritional insights. With SmartRecipe, meal preparation becomes effortless, food waste is reduced, and users are empowered to make smart, delicious, and healthy culinary choices.
Objectives

Food Ingredient Identification: Train a multi-label image classification model that can recognize multiple ingredients present in the given image. Reference Datasets: MAFood121, Foodseg103 https://xiongweiwu.github.io/foodseg103.html


Recipe Recommendation System: Fine-tune an LLM to learn recipe recommendations. The model should analyze the input ingredients and suggest multiple feasible recipes, providing detailed information on ingredient quantities and preparation methods. Reference Dataset: Food Recipe


Nutrient Analysis System: Based on the variety of ingredients in the recipe, associate their nutritional content with the recipe information. When creating the training data for recipes recommendation, this nutritional information can be integrated to provide a more comprehensive recommendation. Reference Dataset: Product Database


# Object1

```js
# 安装必要库
!pip install -q transformers datasets torch torchvision timm
!pip install -q scikit-learn matplotlib seaborn
!pip install -q pillow

```

```python
# 1. Install and import dependencies
!pip install --quiet kaggle torch torchvision pillow matplotlib tqdm

import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm
```

```js

# 4. Download FoodSeg103 dataset
!wget -q https://research.larc.smu.edu.sg/downloads/datarepo/FoodSeg103.zip -P /content/datasets/
!unzip -q -P LARCdataset9947 /content/datasets/FoodSeg103.zip -d /content/datasets
```

```js
# 5. Prepare file lists
image_dir = '/content/datasets/FoodSeg103/images'
mask_dir  = '/content/datasets/FoodSeg103/masks'
image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
mask_paths  = [os.path.join(mask_dir, os.path.basename(p).replace('.jpg', '.png')) for p in image_paths]
```


```js
# 5. Automatically detect image and mask directories
root = '/content/datasets/FoodSeg103'
image_dirs = []
mask_dirs = []
for dirpath, dirnames, filenames in os.walk(root):
    if any(f.lower().endswith('.jpg') for f in filenames):
        image_dirs.append(dirpath)
    if any(f.lower().endswith('.png') for f in filenames):
        mask_dirs.append(dirpath)
if not image_dirs or not mask_dirs:
    raise RuntimeError(f"No image or mask directory found under {root}")
# Use first found directories (adjust index if needed)
image_dir = image_dirs[0]
mask_dir  = mask_dirs[0]
print(f"Using images from: {image_dir}")
print(f"Using masks from:  {mask_dir}")
```
Using images from: /content/datasets/FoodSeg103/Images/img_dir/test

Using masks from:  /content/datasets/FoodSeg103/Images/ann_dir/test

```js
# 6. Prepare paired lists of images and masks
all_images = glob.glob(os.path.join(image_dir, '*.jpg'))
image_paths = []
mask_paths  = []
for img_path in all_images:
    mask_path = os.path.join(mask_dir, os.path.basename(img_path).replace('.jpg', '.png'))
    if os.path.exists(mask_path):
        image_paths.append(img_path)
        mask_paths.append(mask_path)
print(f"Found {len(image_paths)} paired image-mask files.")

if len(image_paths) == 0:
    raise RuntimeError("No image-mask pairs found. Please check your directory structure.")
```

Found 2135 paired image-mask files.
```js
# 7. Determine number of ingredient classes
sample_mask = np.array(Image.open(mask_paths[0]))
num_classes = int(sample_mask.max()) + 1
print(f"Detected {num_classes} ingredient classes.")
```
Detected 97 ingredient classes.

```js
# 8. Dataset and DataLoader
class FoodIngredientDataset(Dataset):
    def __init__(self, img_paths, msk_paths, num_classes, transform=None):
        self.img_paths = img_paths
        self.msk_paths = msk_paths
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        mask = np.array(Image.open(self.msk_paths[idx]))
        labels = np.unique(mask)
        multi_hot = torch.zeros(self.num_classes, dtype=torch.float32)
        for c in labels:
            if c < self.num_classes:
                multi_hot[c] = 1.0
        if self.transform:
            img = self.transform(img)
        return img, multi_hot
```

```js
# 9. Transforms and DataLoader setup
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)
train_dataset = FoodIngredientDataset(train_imgs, train_masks, num_classes, transform=data_transform)
val_dataset   = FoodIngredientDataset(val_imgs,   val_masks,   num_classes, transform=data_transform)
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=2)
val_loader    = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=2)

```
```js
# 10. Model definition
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
```
```js
# Enable CUDA only if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

```js
# 11. Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
```
```js
import torch

num_epochs = 50
# 早停参数设置
patience = 5  # 容忍的停滞epoch数
best_val_loss = float('inf')
counter = 0
best_model_weights = None  # 保存最佳模型参数

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1} Training Loss: {epoch_loss:.4f}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Validation"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}\n")

    # 早停检查
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = model.state_dict()  # 保存最佳模型参数
        counter = 0  # 重置计数器
    else:
        counter += 1
        print(f"Validation loss did not improve. Counter: {counter}/{patience}")
        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}!")
            break

# 训练结束后加载最佳模型参数
model.load_state_dict(best_model_weights)
print("Training completed. Best validation loss: {:.4f}".format(best_val_loss))
```

```js
Epoch 1/50 - Training: 100%|██████████| 54/54 [00:24<00:00,  2.22it/s]
Epoch 1 Training Loss: 0.2449
Validation: 100%|██████████| 14/14 [00:03<00:00,  3.64it/s]
Epoch 1 Validation Loss: 0.1223

Epoch 2/50 - Training: 100%|██████████| 54/54 [00:22<00:00,  2.41it/s]
Epoch 2 Training Loss: 0.1069
Validation: 100%|██████████| 14/14 [00:03<00:00,  3.63it/s]
Epoch 2 Validation Loss: 0.1063

Epoch 3/50 - Training: 100%|██████████| 54/54 [00:23<00:00,  2.33it/s]
Epoch 3 Training Loss: 0.0877
Validation: 100%|██████████| 14/14 [00:03<00:00,  3.63it/s]
Epoch 3 Validation Loss: 0.1017

Epoch 4/50 - Training: 100%|██████████| 54/54 [00:22<00:00,  2.41it/s]
Epoch 4 Training Loss: 0.0726
Validation: 100%|██████████| 14/14 [00:03<00:00,  3.61it/s]
Epoch 4 Validation Loss: 0.0994

Epoch 5/50 - Training: 100%|██████████| 54/54 [00:24<00:00,  2.25it/s]
Epoch 5 Training Loss: 0.0593
Validation: 100%|██████████| 14/14 [00:03<00:00,  3.64it/s]
Epoch 5 Validation Loss: 0.0998

Validation loss did not improve. Counter: 1/5
Epoch 6/50 - Training: 100%|██████████| 54/54 [00:22<00:00,  2.39it/s]
Epoch 6 Training Loss: 0.0492
Validation: 100%|██████████| 14/14 [00:04<00:00,  3.37it/s]
Epoch 6 Validation Loss: 0.0987

Epoch 7/50 - Training: 100%|██████████| 54/54 [00:22<00:00,  2.40it/s]
Epoch 7 Training Loss: 0.0412
Validation: 100%|██████████| 14/14 [00:04<00:00,  2.87it/s]
Epoch 7 Validation Loss: 0.0983

Epoch 8/50 - Training: 100%|██████████| 54/54 [00:22<00:00,  2.45it/s]
Epoch 8 Training Loss: 0.0347
Validation: 100%|██████████| 14/14 [00:04<00:00,  2.86it/s]
Epoch 8 Validation Loss: 0.0997

Validation loss did not improve. Counter: 1/5
Epoch 9/50 - Training: 100%|██████████| 54/54 [00:21<00:00,  2.48it/s]
Epoch 9 Training Loss: 0.0290
Validation: 100%|██████████| 14/14 [00:04<00:00,  3.09it/s]
Epoch 9 Validation Loss: 0.0987

Validation loss did not improve. Counter: 2/5
Epoch 10/50 - Training: 100%|██████████| 54/54 [00:22<00:00,  2.35it/s]
Epoch 10 Training Loss: 0.0246
Validation: 100%|██████████| 14/14 [00:03<00:00,  3.56it/s]
Epoch 10 Validation Loss: 0.1003

Validation loss did not improve. Counter: 3/5
Epoch 11/50 - Training: 100%|██████████| 54/54 [00:22<00:00,  2.36it/s]
Epoch 11 Training Loss: 0.0213
Validation: 100%|██████████| 14/14 [00:03<00:00,  3.51it/s]
Epoch 11 Validation Loss: 0.0997

Validation loss did not improve. Counter: 4/5
Epoch 12/50 - Training: 100%|██████████| 54/54 [00:22<00:00,  2.36it/s]
Epoch 12 Training Loss: 0.0184
Validation: 100%|██████████| 14/14 [00:04<00:00,  3.01it/s]Epoch 12 Validation Loss: 0.1004

Validation loss did not improve. Counter: 5/5
Early stopping triggered at epoch 12!
Training completed. Best validation loss: 0.0983
```


```js
# 13. Save model
os.makedirs('/content/models', exist_ok=True)
torch.save(model.state_dict(), '/content/models/ingredient_classifier1.pth')
print("Model training complete and saved to /content/models/ingredient_classifier1.pth")
```

```js
class_names = [
    "background",
    "candy", "egg tart", "french fries", "chocolate", "biscuit", "popcorn", "pudding", "ice cream", "cheese butter",
    "cake", "wine", "milkshake", "coffee", "juice", "milk", "tea", "almond", "red beans", "cashew",
    "dried cranberries", "soy", "walnut", "peanut", "egg", "apple", "date", "apricot", "avocado", "banana",
    "strawberry", "cherry", "blueberry", "raspberry", "mango", "olives", "peach", "lemon", "pear", "fig",
    "pineapple", "grape", "kiwi", "melon", "orange", "watermelon", "steak", "pork", "chicken duck", "sausage",
    "fried meat", "lamb", "sauce", "crab", "fish", "shellfish", "shrimp", "soup", "bread", "corn", "hamburg",
    "pizza", "hanamaki baozi", "wonton dumplings", "pasta", "noodles", "rice", "pie", "tofu", "eggplant",
    "potato", "garlic", "cauliflower", "tomato", "kelp", "seaweed", "spring onion", "rape", "ginger", "okra",
    "lettuce", "pumpkin", "cucumber", "white radish", "carrot", "asparagus", "bamboo shoots", "broccoli",
    "celery stick", "cilantro mint", "snow peas", "cabbage", "bean sprouts", "onion", "pepper", "green beans",
    "French beans", "king oyster mushroom", "shiitake", "enoki mushroom", "oyster mushroom", "white button mushroom",
    "salad", "other ingredients"
]

```

```js
import matplotlib.pyplot as plt
import os

# 定义标签解码器
def decode_labels(pred, class_names, threshold=0.5):
    return [class_names[i] for i, p in enumerate(pred) if p >= threshold]

# 创建输出文件夹
os.makedirs("/content/results", exist_ok=True)

# 可视化前5张图像
model.eval()
with torch.no_grad():
    for batch_idx, (imgs, labels) in enumerate(val_loader):
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.sigmoid(outputs).detach().cpu().numpy()

        for i in range(min(35, imgs.size(0))):
            img = imgs[i].cpu().permute(1, 2, 0).numpy()
            img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # 反标准化
            img = np.clip(img, 0, 1)

            pred_labels = decode_labels(preds[i], class_names)
            true_labels = decode_labels(labels[i].numpy(), class_names)

            plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.title(f"Predicted: {', '.join(pred_labels)}\nTrue: {', '.join(true_labels)}", fontsize=10)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"/content/results/prediction_{batch_idx}_{i}.png")
            plt.close()
        break  # 只保存第一个 batch 的预测
print("✅ 示例预测结果已保存到 /content/results/")

```
计算 Precision / Recall / F1-score

```js
from sklearn.metrics import precision_score, recall_score, f1_score

y_true, y_pred = [], []

model.eval()
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.sigmoid(outputs).cpu().detach().numpy()
        y_pred.append(preds)
        y_true.append(labels.numpy())

y_pred = np.vstack(y_pred)
y_true = np.vstack(y_true)

# 二值化预测
y_pred_bin = (y_pred >= 0.5).astype(int)

# 每类指标
precision = precision_score(y_true, y_pred_bin, average=None)
recall = recall_score(y_true, y_pred_bin, average=None)
f1 = f1_score(y_true, y_pred_bin, average=None)

for i in range(len(class_names)):
    p = precision[i] if i < len(precision) else 0
    r = recall[i] if i < len(recall) else 0
    f = f1[i] if i < len(f1) else 0
    print(f"{class_names[i]}: Precision={p:.2f}, Recall={r:.2f}, F1={f:.2f}")
```


```js
部分
ea: Precision=0.00, Recall=0.00, F1=0.00
almond: Precision=0.00, Recall=0.00, F1=0.00
red beans: Precision=0.00, Recall=0.00, F1=0.00
cashew: Precision=0.00, Recall=0.00, F1=0.00
dried cranberries: Precision=0.00, Recall=0.00, F1=0.00
soy: Precision=0.00, Recall=0.00, F1=0.00
walnut: Precision=0.00, Recall=0.00, F1=0.00
peanut: Precision=0.00, Recall=0.00, F1=0.00
egg: Precision=0.00, Recall=0.00, F1=0.00
apple: Precision=0.00, Recall=0.00, F1=0.00
date: Precision=0.00, Recall=0.00, F1=0.00
apricot: Precision=0.00, Recall=0.00, F1=0.00
avocado: Precision=0.00, Recall=0.00, F1=0.00
banana: Precision=0.00, Recall=0.00, F1=0.00
strawberry: Precision=0.96, Recall=0.69, F1=0.80
cherry: Precision=0.00, Recall=0.00, F1=0.00
blueberry: Precision=1.00, Recall=0.46, F1=0.63
raspberry: Precision=0.00, Recall=0.00, F1=0.00
mango: Precision=0.00, Recall=0.00, F1=0.00
olives: Precision=0.00, Recall=0.00, F1=0.00
```
混淆矩阵（多标签版本）+ AUC

```js
from sklearn.metrics import multilabel_confusion_matrix, roc_auc_score

# 混淆矩阵
mcm = multilabel_confusion_matrix(y_true, y_pred_bin)
for i, cm in enumerate(mcm):
    tn, fp, fn, tp = cm.ravel()
    print(f"{class_names[i]}: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

# 多标签 AUC
try:
    auc = roc_auc_score(y_true, y_pred, average="macro")
    print(f"Macro AUC: {auc:.4f}")
except ValueError as e:
    print("⚠️ AUC 计算失败：可能是某些类全为0")

```

```js
background: TP=427, FP=0, FN=0, TN=0
candy: TP=0, FP=0, FN=1, TN=426
egg tart: TP=0, FP=0, FN=0, TN=427
french fries: TP=0, FP=1, FN=13, TN=413
chocolate: TP=0, FP=0, FN=2, TN=425
biscuit: TP=1, FP=0, FN=19, TN=407
popcorn: TP=0, FP=0, FN=0, TN=427
pudding: TP=0, FP=0, FN=1, TN=426
```
```js
model.eval()

# 获取一批验证数据
imgs, labels = next(iter(val_loader))
imgs, labels = imgs.to(device), labels.to(device)

# 预测结果
with torch.no_grad():
    outputs = model(imgs)
    preds = torch.sigmoid(outputs).detach().cpu().numpy()

# 动态获取批次大小并限制显示数量
batch_size = imgs.size(0)  # 实际批次大小
max_show = 3  # 最多显示3张
for i in range(min(max_show, batch_size)):
    # 反归一化图像
    img = imgs[i].cpu().permute(1, 2, 0).numpy()  # (H, W, C)
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)

    # 解码预测和真实标签
    pred_labels = decode_labels(preds[i], class_names)  # 确保decode_labels函数存在
    true_labels = decode_labels(labels[i].cpu().numpy(), class_names)

    # 可视化
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(f"Predicted: {', '.join(pred_labels)}\nTrue: {', '.join(true_labels)}")
    plt.axis("off")
    plt.show()
```

1. 添加预测测试与可视化结果
   
展示模型预测的实际效果（图片 + 预测标签 vs. 真实标签）


```js
import torch
import numpy as np
import matplotlib.pyplot as plt

# 确保模型处于评估模式
model.eval()

# 获取一批验证数据（自动适配当前batch_size）
imgs, labels = next(iter(val_loader))
imgs, labels = imgs.to(device), labels.to(device)

# 预测结果
with torch.no_grad():
    outputs = model(imgs)
    preds = torch.sigmoid(outputs).detach().cpu().numpy()  # 多标签分类预测

# 动态获取批次大小
batch_size = imgs.size(0)  # 实际当前批次大小
max_show = 30  # 最多显示3张图片

# 显示预测结果
for i in range(min(max_show, batch_size)):
    # --- 图像处理部分 ---
    # 将张量转为numpy并调整维度顺序 (C, H, W) -> (H, W, C)
    img = imgs[i].cpu().permute(1, 2, 0).numpy()

    # 反归一化（假设使用ImageNet均值和标准差）
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)  # 确保像素值在[0,1]范围内

    # --- 标签处理部分 ---
    # 假设 decode_labels 函数将概率转换为类别名称
    # preds[i] 应该是形状为 (num_classes,) 的numpy数组
    pred_labels = decode_labels(preds[i], class_names)  # 自定义解码函数

    # 真实标签需要从tensor转换
    true_labels = decode_labels(labels[i].cpu().numpy(), class_names)

    # --- 可视化部分 ---
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(f"Predicted: {', '.join(pred_labels)}\nTrue: {', '.join(true_labels)}")
    plt.axis("off")
    plt.show()
```
2. 评估指标报告（精确率、召回率、F1 分数）

   
使用 sklearn.metrics 对模型性能进行全面评估：

```js
from sklearn.metrics import precision_score, recall_score, f1_score

y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.sigmoid(outputs).cpu().numpy()
        y_pred.extend((preds > 0.5).astype(int))
        y_true.extend(labels.numpy())

print("Precision:", precision_score(y_true, y_pred, average='micro'))
print("Recall:", recall_score(y_true, y_pred, average='micro'))
print("F1 Score:", f1_score(y_true, y_pred, average='micro'))

```

```js
Precision: 0.8333333333333334
Recall: 0.424886191198786
F1 Score: 0.5628140703517588
```
```js
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

# 确保模型处于评估模式
model.eval()

# 初始化存储所有预测和真实标签的容器
all_preds = []
all_labels = []

# 遍历验证集获取所有预测结果
with torch.no_grad():
    for imgs, labels in tqdm(val_loader, desc="Collecting predictions"):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

# 合并所有批次的预测和标签
all_preds = np.concatenate(all_preds, axis=0)  # 形状 (num_samples, num_classes)
all_labels = np.concatenate(all_labels, axis=0)  # 形状 (num_samples, num_classes)

# 计算每个类别的ROC曲线和AUC
num_classes = all_labels.shape[1]
plt.figure(figsize=(10, 8))

for class_idx in range(num_classes):
    # 获取当前类别的真实标签和预测概率
    y_true = all_labels[:, class_idx]
    y_score = all_preds[:, class_idx]

    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # 绘制曲线
    plt.plot(fpr, tpr, lw=2,
             label=f'Class {class_idx} ({class_names[class_idx]}) AUC = {roc_auc:.3f}')

# 绘制对角线参考线
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves per Class')
plt.legend(loc="lower right", fontsize=8)
plt.show()
```

