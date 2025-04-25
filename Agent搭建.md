为了帮助您在 Google Colab 上实现 SmartRecipe 项目的三个核心功能（食材识别、食谱推荐和营养分析），我整理了基于现有开源模型的完整解决方案。以下是每个模块的详细说明及其对应的代码示例：

---

## 🥦 模块 1：食材识别（图像识别）

我们将使用预训练的 DenseNet121 模型进行食材识别。该模型在 Food101 数据集上表现良好，适合多标签分类任务。 ([GitHub - kama34/food-image-recognition: This project explores food image classification using baseline convolutional neural networks and transfer learning with DENSENET121](https://github.com/kama34/food-image-recognition?utm_source=chatgpt.com))

### 🔧 安装依赖


```python
!pip install torch torchvision
```


### 📥 下载并加载模型


```python
import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# 加载预训练的 DenseNet121 模型
model = models.densenet121(pretrained=True)
model.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 加载并预处理图像
def load_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    return batch_t

# 示例图像 URL
image_url = 'https://example.com/your_image.jpg'
input_tensor = load_image(image_url)

# 进行预测
with torch.no_grad():
    output = model(input_tensor)
    _, indices = torch.topk(output, 5)
    print(indices)
```


请将 `image_url` 替换为您自己的图像链接。该代码将输出预测的前5个类别索引，您可以根据实际的类别标签映射来获取对应的食材名称。

---

## 🍳 模块 2：食谱推荐系统（基于 LLM）

我们将使用 OpenAI 的 GPT-3.5 模型，通过 LangChain 库进行食谱推荐。用户输入食材列表，模型将返回详细的食谱建议。 ([Recipe Recommendation System Using Python - GeeksforGeeks](https://www.geeksforgeeks.org/recipe-recommendation-system-using-python/?utm_source=chatgpt.com))

### 🔧 安装依赖


```python
!pip install openai langchain streamlit
```


### 🧠 实现食谱推荐


```python
import openai
from langchain.llms import OpenAI

# 设置 OpenAI API 密钥
openai.api_key = 'your-api-key'

# 初始化模型
llm = OpenAI(temperature=0.7)

# 用户输入的食材列表
ingredients = "鸡蛋, 西红柿, 洋葱"

# 构建提示语
prompt = f"我有以下食材：{ingredients}。请推荐一个简单的食谱，包括所需食材的数量和详细的制作步骤。"

# 获取模型回复
response = llm(prompt)
print(response)
```


请将 `'your-api-key'` 替换为您自己的 OpenAI API 密钥。该代码将输出一个包含所需食材数量和详细制作步骤的食谱。 ([Recipe Recommendation System Using Python - GeeksforGeeks](https://www.geeksforgeeks.org/recipe-recommendation-system-using-python/?utm_source=chatgpt.com))

---

## 🧮 模块 3：营养分析系统

我们将使用 OpenFoodFacts 数据集，通过其提供的 API 获取食材的营养信息，并计算整个食谱的营养成分。

### 🔧 安装依赖


```python
!pip install requests
```


### 🥗 实现营养分析


```python
import requests

def get_nutrition(ingredient):
    url = f"https://world.openfoodfacts.org/api/v0/product/{ingredient}.json"
    response = requests.get(url)
    data = response.json()
    if 'product' in data and 'nutriments' in data['product']:
        nutriments = data['product']['nutriments']
        return {
            'energy': nutriments.get('energy-kcal_100g', 0),
            'proteins': nutriments.get('proteins_100g', 0),
            'fat': nutriments.get('fat_100g', 0),
            'carbohydrates': nutriments.get('carbohydrates_100g', 0)
        }
    else:
        return {
            'energy': 0,
            'proteins': 0,
            'fat': 0,
            'carbohydrates': 0
        }

# 示例食材列表
ingredients = ['egg', 'tomato', 'onion']

# 计算总营养成分
total_nutrition = {'energy': 0, 'proteins': 0, 'fat': 0, 'carbohydrates': 0}
for item in ingredients:
    nutrition = get_nutrition(item)
    for key in total_nutrition:
        total_nutrition[key] += nutrition[key]

print("总营养成分（每100g）：")
print(total_nutrition)
```


该代码将输出所提供食材的总营养成分，包括每100克的能量、蛋白质、脂肪和碳水化合物含量。

---

## ✅ 总结

通过上述三个模块的实现，您可以在 Google Colab 上构建一个完整的 SmartRecipe 应用程序，涵盖食材识别、食谱推荐和营养分析功能。

如果您需要进一步的帮助，例如将这些模块集成到一个完整的应用程序中，或是部署到 Web 界面上，请随时告诉我！ 
