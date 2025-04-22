为了进一步增强你当前的代码，以下是几个改进建议及其对应的代码实现，能够在 Google Colab 上运行：

### 1. **错误处理和日志记录**
在数据集处理或模型训练时，出现错误的可能性较大，加入错误处理机制可以帮助更好地调试和运行代码。

#### 错误处理和日志记录
```python
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# 示例：在数据准备时添加错误处理
try:
    # 加载数据集
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    mask_paths = [os.path.join(mask_dir, os.path.basename(p).replace('.jpg', '.png')) for p in image_paths]
    if len(image_paths) == 0:
        raise ValueError("没有找到任何图片文件。请检查路径。")
    logger.info(f"Found {len(image_paths)} image files and corresponding masks.")
except Exception as e:
    logger.error(f"Error while preparing dataset: {e}")
    raise
```

### 2. **保存和加载模型**
保存模型时，可以考虑为其提供一个版本号或者时间戳，以便跟踪不同版本的模型。同时，加载模型时可以确保路径存在。

#### 保存和加载模型
```python
import time

# 保存模型时，加入时间戳，便于管理
model_save_path = f'/content/models/ingredient_classifier_{int(time.time())}.pth'
torch.save(model.state_dict(), model_save_path)
logger.info(f"模型已保存至 {model_save_path}")

# 加载模型
def load_model(model, model_path):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        logger.info(f"模型已加载从 {model_path}")
    else:
        logger.error("模型路径不存在，请检查路径")
    return model

# 示例：加载之前保存的模型
model = load_model(model, model_save_path)
```

### 3. **更多评估指标和混淆矩阵**
除了 `precision`、`recall` 和 `F1` 分数外，还可以加入混淆矩阵和 ROC 曲线来进一步评估模型的表现。以下是如何实现混淆矩阵和 ROC 曲线的代码：

#### 添加混淆矩阵和ROC曲线
```python
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# 混淆矩阵
def plot_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels.flatten(), preds.flatten())
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# ROC曲线
def plot_roc_curve(labels, preds, class_names):
    fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

# 计算并绘制评估图
def evaluate_with_metrics(model, dataloader, class_names, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Evaluating"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.sigmoid(outputs).cpu().numpy()
            labels = labels.numpy()
            
            all_preds.append(preds)
            all_labels.append(labels)
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # 混淆矩阵
    binary_preds = (all_preds >= threshold).astype(int)
    plot_confusion_matrix(all_labels, binary_preds, class_names)

    # ROC曲线
    plot_roc_curve(all_labels, all_preds, class_names)

    # 打印其他指标
    precision = precision_score(all_labels, binary_preds, average='micro')
    recall = recall_score(all_labels, binary_preds, average='micro')
    f1 = f1_score(all_labels, binary_preds, average='micro')
    
    try:
        auc = roc_auc_score(all_labels, all_preds, average='micro')
    except:
        auc = 'N/A (AUC not defined for this batch)'
    
    print("🔍 Evaluation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc}")

# 使用方式
evaluate_with_metrics(model, val_loader, class_names)
```

### 4. **动态调整学习率**
在训练过程中，可以使用 `lr_scheduler` 来动态调整学习率，这样有助于提升模型性能。

#### 动态学习率调整
```python
from torch.optim.lr_scheduler import StepLR

# 定义学习率调度器
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# 在训练循环中使用
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
    
    # 每个 epoch 后更新学习率
    scheduler.step()
    
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
```

### 总结
- 添加了日志记录和错误处理，以便更好地跟踪和调试代码。
- 提供了保存和加载模型的代码，确保可以在不同阶段加载模型进行测试。
- 增加了更多的评估指标，如混淆矩阵和ROC曲线，帮助你更全面地评估模型。
- 引入了动态调整学习率的机制，以优化训练过程。

这些代码改进可以让你在 Google Colab 上进行更加高效和可靠的训练和评估。
