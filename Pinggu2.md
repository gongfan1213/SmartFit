好的！下面我给你整理两部分代码：  
1. ✅ **多标签分类评估函数（Precision / Recall / F1 / AUC）**  
2. ✅ **Grad-CAM 热力图可视化（解释模型在图像上看的区域）**

这两段代码可以**直接复制粘贴到你的 Colab 的末尾**，立刻运行。

---

## ✅ 1. 评估函数（Precision / Recall / F1 / AUC）

> 💡 建议在训练完成后调用，用于验证集评估。

```python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, dataloader, threshold=0.5):
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

    binary_preds = (all_preds >= threshold).astype(int)

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

# 👉 使用方式：
evaluate_model(model, val_loader)
```

---

## ✅ 2. Grad-CAM 可解释性可视化（展示模型“看哪里”）

> 💡 可视化模型关注的区域，展示在 demo 或报告里有加分项！

```python
!pip install --quiet grad-cam

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())

# 可视化验证集第一张图
model.eval()
with torch.no_grad():
    imgs, labels = next(iter(val_loader))
    img = imgs[0].unsqueeze(0).to(device)
    label = labels[0].cpu().numpy()

    outputs = model(img)
    probs = torch.sigmoid(outputs).cpu().numpy()[0]
    top_idx = np.argmax(probs)  # 获取概率最高的类别

    grayscale_cam = cam(input_tensor=img, targets=[ClassifierOutputTarget(top_idx)])
    grayscale_cam = grayscale_cam[0, :]
    img_np = img[0].cpu().permute(1, 2, 0).numpy()
    img_np = img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    img_np = np.clip(img_np, 0, 1)

    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    plt.figure(figsize=(6,6))
    plt.imshow(cam_image)
    plt.title(f"Grad-CAM for predicted class: {class_names[top_idx]}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
```

---

## 🔚 建议放置顺序

你可以把这两个模块分别放在：

- 📍 `model training complete` 输出之后（评估函数）
- 📍 `prediction 可视化保存` 之后（Grad-CAM）

---

要是你希望我把整个整理好的代码模块贴成一个整体脚本版本，我也可以给你打包好，你就可以一键复制粘贴。需要我这样做吗？
