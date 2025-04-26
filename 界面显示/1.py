# 安装依赖库
!pip install git+https://github.com/openai/CLIP.git -q
!pip install transformers -q
!pip install openfoodfacts -q
!pip install torchvision matplotlib -q

import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from transformers import pipeline
import openfoodfacts

# 设置随机种子保证可重复性
torch.manual_seed(42)

# ------------------
# 食材识别模块
# ------------------
def image_recognition():
    # 加载CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # 定义食材候选列表（可根据需要扩展）
    ingredient_list = [
        "apple", "banana", "carrot", "broccoli", "tomato",
        "potato", "onion", "chicken", "egg", "flour",
        "milk", "cheese", "fish", "beef", "sugar"
    ]
    
    # 加载测试图片（这里使用示例图片，可替换为任意图片URL）
    image_url = "https://storage.googleapis.com/sfr-vision-language-research/CLIP/benchmark.jpg"  # 示例图片包含苹果和胡萝卜
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    
    # 预处理和模型推理
    image = preprocess(img).unsqueeze(0).to(device)
    text = clip.tokenize([f"a photo of {ingredient}" for ingredient in ingredient_list]).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    
    # 获取预测结果
    top_k = 3
    top_indices = probs.argsort()[0][-top_k:][::-1]
    predicted_ingredients = [ingredient_list[i] for i in top_indices]
    
    # 显示结果（假设真实标签为已知）
    true_ingredients = ["apple", "carrot"]  # 示例图片真实标签
    
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {', '.join(predicted_ingredients)}\nTrue: {', '.join(true_ingredients)}")
    plt.show()
    
    return predicted_ingredients

# ------------------
# 食谱推荐模块
# ------------------
def recipe_recommendation(ingredients):
    generator = pipeline('text-generation', model='gpt2-medium')
    
    prompt = f"Create a detailed recipe using {', '.join(ingredients)}:\n1."
    recipe = generator(
        prompt,
        max_length=300,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9
    )
    
    print("\nGenerated Recipe:")
    print(recipe[0]['generated_text'])

# ------------------
# 营养分析模块
# ------------------
def nutritional_analysis(ingredients):
    print("\nNutritional Analysis (per 100g):")
    
    nutrition = {
        'energy': 0,
        'fat': 0,
        'carbohydrates': 0,
        'proteins': 0
    }
    
    for ingredient in ingredients:
        try:
            products = openfoodfacts.products.search(ingredient)
            if products['products']:
                product = products['products'][0]
                nutriments = product.get('nutriments', {})
                
                nutrition['energy'] += nutriments.get('energy-kcal_100g', 0) or 0
                nutrition['fat'] += nutriments.get('fat_100g', 0) or 0
                nutrition['carbohydrates'] += nutriments.get('carbohydrates_100g', 0) or 0
                nutrition['proteins'] += nutriments.get('proteins_100g', 0) or 0
        except:
            continue
    
    print(f"Calories: {nutrition['energy']:.1f} kcal")
    print(f"Fat: {nutrition['fat']:.1f} g")
    print(f"Carbohydrates: {nutrition['carbohydrates']:.1f} g")
    print(f"Proteins: {nutrition['proteins']:.1f} g")

# ------------------
# 主程序
# ------------------
if __name__ == "__main__":
    # 执行图像识别
    detected_ingredients = image_recognition()
    
    # 食谱推荐
    recipe_recommendation(detected_ingredients)
    
    # 营养分析
    nutritional_analysis(detected_ingredients)