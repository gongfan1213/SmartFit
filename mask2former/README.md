
# FoodSeg103：Mask2Former在语义分割上的微调 🍔🍕

## 项目概述

本项目专注于在**FoodSeg103**数据集上对Mask2Former模型进行语义分割微调。目标是提升模型在识别和分割各类食物图像中的表现。项目还包括模型部署，并使用Gradio创建了一个用户友好的GUI，实现交互式推理。

### 🎥 演示

下方GIF展示了Gradio界面的实际效果。🍴✨

<div align="center">
  <img src="https://raw.githubusercontent.com/NimaVahdat/FoodSeg_mask2former/main/demo.gif">
</div>

## 🚀 快速开始

### 安装

1. **克隆仓库：**
   ```bash
   git clone https://github.com/NimaVahdat/FoodSeg_mask2former.git
   cd FoodSeg_mask2former
   ```

2. **安装依赖：**
   ```bash
   pip install -r requirements.txt
   ```

### 配置

在 `config.yaml` 文件中配置训练参数：

- `batch_size`：每个批次的样本数量。
- `learning_rate`：优化器的初始学习率。
- `step_size`：学习率调整的周期（以epoch为单位）。
- `gamma`：学习率衰减因子。
- `epochs`：训练总轮数。
- `save_path`：模型检查点保存目录。
- `load_checkpoint`：预训练检查点路径（或设为 `None` 从头训练）。
- `log_dir`：TensorBoard日志目录。

### 训练

启动训练流程，执行：
```bash
python  -m scripts.run_training
```
该命令会根据 `config.yaml` 中的参数初始化训练，并将训练好的模型检查点保存到指定的 `save_path`。

### 使用Gradio部署模型

通过Gradio部署模型，创建一个交互式Web界面，用户可以上传图片并实时查看分割结果。

1. **运行Gradio应用：**
   ```bash
   python -m gradio_app.app
   ```

2. **访问界面：**
   打开浏览器，访问终端中提供的URL，即可开始与模型交互。

## 模型与数据集

### Mask2Former模型
- **Mask2Former** 是一款面向实例分割和语义分割任务的先进模型。它采用基于Transformer的架构，能够提供准确且鲁棒的分割结果。
- 本项目对Mask2Former进行了FoodSeg103数据集的微调，使其更适应食物相关的分割任务。

### FoodSeg103数据集
- [**FoodSeg103**](https://huggingface.co/datasets/EduardoPacheco/FoodSeg103) 是一个包含103种食物类别的综合语义分割数据集，提供了多样化且带注释的食物图像，用于训练和评估分割模型。

## 结果

- **平均交并比（mIoU）**：在验证集上取得了 **4.21** 的mIoU分数。通过提升计算资源和延长微调时间，模型表现有望进一步提升。

## 📚 许可证
- **许可协议：** 本项目采用MIT许可证。详情请参见 [LICENSE](LICENSE) 文件。

## 📞 联系方式

如有问题、反馈或贡献建议，请提交Issue或与我联系。
```
如需进一步润色或有其他需求，请随时告知！