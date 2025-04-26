import streamlit as st
import torch
import clip
from PIL import Image
from transformers import pipeline

# 设置随机种子保证可重复性
torch.manual_seed(42)

# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 定义食材候选列表
ingredient_list = [
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

nutrition_db = {
        # 甜品类
        "candy": {"energy": 380, "fat": 0.1, "carbohydrates": 93, "protein": 0.2},
        "egg tart": {"energy": 290, "fat": 17, "carbohydrates": 31, "protein": 5},
        "french fries": {"energy": 312, "fat": 15, "carbohydrates": 41, "protein": 3},
        "chocolate": {"energy": 546, "fat": 31, "carbohydrates": 61, "protein": 5},
        "biscuit": {"energy": 460, "fat": 18, "carbohydrates": 68, "protein": 6},
        "popcorn": {"energy": 387, "fat": 4.5, "carbohydrates": 78, "protein": 12},
        "pudding": {"energy": 130, "fat": 4, "carbohydrates": 20, "protein": 3},
        "ice cream": {"energy": 207, "fat": 11, "carbohydrates": 24, "protein": 3.5},
        "cheese butter": {"energy": 717, "fat": 81, "carbohydrates": 0.6, "protein": 0.9},
        "cake": {"energy": 347, "fat": 15, "carbohydrates": 53, "protein": 5},
        # 饮品类
        "wine": {"energy": 83, "fat": 0, "carbohydrates": 2.6, "protein": 0.1},
        "milkshake": {"energy": 112, "fat": 3, "carbohydrates": 18, "protein": 3.3},
        "coffee": {"energy": 1, "fat": 0, "carbohydrates": 0.2, "protein": 0.1},
        "juice": {"energy": 46, "fat": 0.1, "carbohydrates": 11, "protein": 0.3},
        "milk": {"energy": 42, "fat": 1.0, "carbohydrates": 5.0, "protein": 3.4},
        "tea": {"energy": 1, "fat": 0, "carbohydrates": 0.3, "protein": 0},
        # 坚果类
        "almond": {"energy": 579, "fat": 50, "carbohydrates": 22, "protein": 21},
        "cashew": {"energy": 553, "fat": 44, "carbohydrates": 30, "protein": 18},
        "walnut": {"energy": 654, "fat": 65, "carbohydrates": 14, "protein": 15},
        "peanut": {"energy": 567, "fat": 49, "carbohydrates": 16, "protein": 26},
        "dried cranberries": {"energy": 308, "fat": 1.4, "carbohydrates": 82, "protein": 0.2},
        # 蛋奶制品
        "egg": {"energy": 143, "fat": 9.5, "carbohydrates": 0.7, "protein": 12.6},
        "soy": {"energy": 446, "fat": 20, "carbohydrates": 30, "protein": 36},
        # 水果类
        "apple": {"energy": 52, "fat": 0.2, "carbohydrates": 14, "protein": 0.3},
        "date": {"energy": 282, "fat": 0.4, "carbohydrates": 75, "protein": 2.5},
        "apricot": {"energy": 48, "fat": 0.4, "carbohydrates": 11, "protein": 1.4},
        "avocado": {"energy": 160, "fat": 15, "carbohydrates": 9, "protein": 2},
        "banana": {"energy": 89, "fat": 0.3, "carbohydrates": 23, "protein": 1.1},
        "strawberry": {"energy": 32, "fat": 0.3, "carbohydrates": 7.7, "protein": 0.7},
        "cherry": {"energy": 50, "fat": 0.3, "carbohydrates": 12, "protein": 1},
        "blueberry": {"energy": 57, "fat": 0.3, "carbohydrates": 14, "protein": 0.7},
        "raspberry": {"energy": 52, "fat": 0.7, "carbohydrates": 12, "protein": 1.2},
        "mango": {"energy": 60, "fat": 0.4, "carbohydrates": 15, "protein": 0.8},
        "olives": {"energy": 115, "fat": 11, "carbohydrates": 6, "protein": 0.8},
        "peach": {"energy": 39, "fat": 0.3, "carbohydrates": 10, "protein": 0.9},
        "lemon": {"energy": 29, "fat": 0.3, "carbohydrates": 9, "protein": 1.1},
        "pear": {"energy": 57, "fat": 0.1, "carbohydrates": 15, "protein": 0.4},
        "fig": {"energy": 74, "fat": 0.3, "carbohydrates": 19, "protein": 0.8},
        "pineapple": {"energy": 50, "fat": 0.1, "carbohydrates": 13, "protein": 0.5},
        "grape": {"energy": 69, "fat": 0.2, "carbohydrates": 18, "protein": 0.7},
        "kiwi": {"energy": 61, "fat": 0.5, "carbohydrates": 15, "protein": 1.1},
        "melon": {"energy": 34, "fat": 0.2, "carbohydrates": 8, "protein": 0.8},
        "orange": {"energy": 47, "fat": 0.1, "carbohydrates": 12, "protein": 0.9},
        "watermelon": {"energy": 30, "fat": 0.2, "carbohydrates": 8, "protein": 0.6},
        # 肉类
        "steak": {"energy": 271, "fat": 19, "carbohydrates": 0, "protein": 25},
        "pork": {"energy": 242, "fat": 14, "carbohydrates": 0, "protein": 27},
        "chicken duck": {"energy": 239, "fat": 13, "carbohydrates": 0, "protein": 27},
        "sausage": {"energy": 346, "fat": 31, "carbohydrates": 1.5, "protein": 14},
        "fried meat": {"energy": 280, "fat": 18, "carbohydrates": 0, "protein": 26},
        "lamb": {"energy": 294, "fat": 21, "carbohydrates": 0, "protein": 25},
        # 海鲜类
        "crab": {"energy": 83, "fat": 0.7, "carbohydrates": 0, "protein": 18},
        "fish": {"energy": 206, "fat": 12, "carbohydrates": 0, "protein": 22},
        "shellfish": {"energy": 85, "fat": 0.5, "carbohydrates": 2, "protein": 18},
        "shrimp": {"energy": 99, "fat": 0.3, "carbohydrates": 0.2, "protein": 24},
        # 主食类
        "bread": {"energy": 265, "fat": 3.2, "carbohydrates": 49, "protein": 9},
        "corn": {"energy": 86, "fat": 1.4, "carbohydrates": 19, "protein": 3.3},
        "hamburg": {"energy": 295, "fat": 14, "carbohydrates": 30, "protein": 15},
        "pizza": {"energy": 266, "fat": 10, "carbohydrates": 33, "protein": 11},
        "hanamaki baozi": {"energy": 180, "fat": 2, "carbohydrates": 35, "protein": 6},
        "wonton dumplings": {"energy": 150, "fat": 5, "carbohydrates": 20, "protein": 8},
        "pasta": {"energy": 131, "fat": 1.1, "carbohydrates": 25, "protein": 5},
        "noodles": {"energy": 138, "fat": 2.1, "carbohydrates": 25, "protein": 4.5},
        "rice": {"energy": 130, "fat": 0.3, "carbohydrates": 28, "protein": 2.7},
        "pie": {"energy": 260, "fat": 14, "carbohydrates": 31, "protein": 3},
        # 蔬菜类
        "tofu": {"energy": 76, "fat": 4.8, "carbohydrates": 1.9, "protein": 8},
        "eggplant": {"energy": 25, "fat": 0.2, "carbohydrates": 6, "protein": 1},
        "potato": {"energy": 77, "fat": 0.1, "carbohydrates": 17, "protein": 2},
        "garlic": {"energy": 149, "fat": 0.5, "carbohydrates": 33, "protein": 6},
        "cauliflower": {"energy": 25, "fat": 0.3, "carbohydrates": 5, "protein": 2},
        "tomato": {"energy": 18, "fat": 0.2, "carbohydrates": 3.9, "protein": 0.9},
        "kelp": {"energy": 43, "fat": 0.6, "carbohydrates": 9, "protein": 1.7},
        "seaweed": {"energy": 35, "fat": 0.3, "carbohydrates": 5, "protein": 5},
        "spring onion": {"energy": 32, "fat": 0.2, "carbohydrates": 7, "protein": 1.8},
        "rape": {"energy": 25, "fat": 0.1, "carbohydrates": 3.5, "protein": 2.5},
        "ginger": {"energy": 80, "fat": 0.8, "carbohydrates": 18, "protein": 1.8},
        "okra": {"energy": 33, "fat": 0.2, "carbohydrates": 7, "protein": 2},
        "lettuce": {"energy": 15, "fat": 0.2, "carbohydrates": 2.9, "protein": 1.4},
        "pumpkin": {"energy": 26, "fat": 0.1, "carbohydrates": 7, "protein": 1},
        "cucumber": {"energy": 15, "fat": 0.1, "carbohydrates": 3.6, "protein": 0.7},
        "white radish": {"energy": 16, "fat": 0.1, "carbohydrates": 3.4, "protein": 0.7},
        "carrot": {"energy": 41, "fat": 0.2, "carbohydrates": 10, "protein": 0.9},
        "asparagus": {"energy": 20, "fat": 0.1, "carbohydrates": 3.9, "protein": 2.2},
        "bamboo shoots": {"energy": 27, "fat": 0.3, "carbohydrates": 5, "protein": 2.6},
        "broccoli": {"energy": 34, "fat": 0.4, "carbohydrates": 6.6, "protein": 2.8},
        "celery stick": {"energy": 16, "fat": 0.2, "carbohydrates": 3, "protein": 0.7},
        "cilantro mint": {"energy": 23, "fat": 0.5, "carbohydrates": 3.7, "protein": 2.1},
        "snow peas": {"energy": 42, "fat": 0.2, "carbohydrates": 7.5, "protein": 2.8},
        "cabbage": {"energy": 25, "fat": 0.1, "carbohydrates": 6, "protein": 1.3},
        "bean sprouts": {"energy": 30, "fat": 0.2, "carbohydrates": 5.9, "protein": 3},
        "onion": {"energy": 40, "fat": 0.1, "carbohydrates": 9, "protein": 1.1},
        "pepper": {"energy": 20, "fat": 0.2, "carbohydrates": 4.6, "protein": 0.9},
        "green beans": {"energy": 31, "fat": 0.2, "carbohydrates": 7, "protein": 1.8},
        "French beans": {"energy": 31, "fat": 0.1, "carbohydrates": 7, "protein": 1.8},
        # 菌菇类
        "king oyster mushroom": {"energy": 35, "fat": 0.5, "carbohydrates": 6, "protein": 3},
        "shiitake": {"energy": 34, "fat": 0.5, "carbohydrates": 7, "protein": 2.2},
        "enoki mushroom": {"energy": 37, "fat": 0.3, "carbohydrates": 8, "protein": 2.7},
        "oyster mushroom": {"energy": 33, "fat": 0.4, "carbohydrates": 6, "protein": 3.3},
        "white button mushroom": {"energy": 22, "fat": 0.3, "carbohydrates": 3.3, "protein": 3.1},
        # 其他
        "sauce": {"energy": 120, "fat": 10, "carbohydrates": 8, "protein": 1},
        "soup": {"energy": 35, "fat": 1.5, "carbohydrates": 4, "protein": 2},
        "salad": {"energy": 50, "fat": 3, "carbohydrates": 5, "protein": 2},
        "other ingredients": {"energy": 100, "fat": 5, "carbohydrates": 12, "protein": 3}
    }

def image_recognition(img):
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
    
    return predicted_ingredients

def recipe_recommendation(ingredients):
    generator = pipeline('text-generation', model='gpt2-medium')
    
    prompt = f"Create a detailed recipe using {', '.join(ingredients)} with exact quantities and step-by-step instructions:\n1. Ingredients:\n"
    for i, ing in enumerate(ingredients, 1):
        prompt += f"{i}. {ing}: [quantity]\n"
    prompt += "\n2. Instructions:\n1."
    
    recipe = generator(
        prompt,
        max_length=500,
        num_return_sequences=1,
        temperature=0.6,
        top_p=0.85,
        do_sample=True
    )
    
    return recipe[0]['generated_text']

def nutritional_analysis(ingredients):
    # 扩展的默认营养数据库（每100克）
    
    
    total = {"energy": 0, "fat": 0, "carbohydrates": 0, "protein": 0}
    
    for ingredient in ingredients:
        if ingredient.lower() in nutrition_db:
            data = nutrition_db[ingredient.lower()]
            total["energy"] += data["energy"]
            total["fat"] += data["fat"]
            total["carbohydrates"] += data["carbohydrates"]
            total["protein"] += data["protein"]
        else:
            st.warning(f"⚠️ No data for {ingredient} (using placeholder values)")
            # 使用默认平均值作为占位值
            total["energy"] += 150
            total["fat"] += 5
            total["carbohydrates"] += 15
            total["protein"] += 5
    
    return total

def main():
    st.title("🍳 Smart Recipe Recommendation System")
    st.write("Upload an image of ingredients to get recipe suggestions and nutritional analysis")
    # 新增拍照上传功能
    show_camera = st.session_state.get('show_camera', False)
    if st.button("Take a photo to Upload"):
        st.session_state['show_camera'] = True
        show_camera = True
    camera_image = None
    if show_camera:
        camera_image = st.camera_input("Take a photo...")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    img = None
    if camera_image is not None:
        img = Image.open(camera_image)
        st.image(img, caption='📸 Captured Photo', use_column_width=True)
    elif uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='📸 Uploaded Image', use_column_width=True)
    if img is not None:
        st.markdown("<hr style='border: 2px solid #FF4B4B;'>", unsafe_allow_html=True)
        # 食材识别
        detected_ingredients = image_recognition(img)
        st.subheader("🔍 Detected Ingredients")
        st.markdown(f"<span style='font-size:22px;color:#4B8BBE;'>🥗 <b>{', '.join(detected_ingredients).title()}</b></span>", unsafe_allow_html=True)
        st.info("These are the main ingredients I found! 🍅🥚🥩")
        st.markdown("<hr style='border: 1px dashed #4B8BBE;'>", unsafe_allow_html=True)
        # 食谱推荐
        st.subheader("🍽️ Recommended Recipes")
        recipe = recipe_recommendation(detected_ingredients)
        st.markdown(f"<div style='background:#FFF3E0;padding:16px;border-radius:12px;border:1px solid #FFD180;'><b>Chef's Suggestion 👨‍🍳:</b><br><span style='font-size:18px;'>{recipe}</span></div>", unsafe_allow_html=True)
        st.success("Try these delicious dishes! 😋")
        st.markdown("<hr style='border: 1px dashed #FFD180;'>", unsafe_allow_html=True)
        # 营养分析
        st.subheader("📊 Nutritional Analysis")
        nutrition = nutritional_analysis(detected_ingredients)
        st.write("### 🧾 Detailed Nutrition per 100g")
        nutrition_table = []
        for ing in detected_ingredients:
            if ing.lower() in nutrition_db:
                data = nutrition_db[ing.lower()]
                nutrition_table.append({
                    "Ingredient": ing.title(),
                    "Calories": data["energy"],
                    "Fat (g)": data["fat"],
                    "Carbs (g)": data["carbohydrates"],
                    "Protein (g)": data["protein"]
                })
        if nutrition_table:
            st.dataframe(nutrition_table, use_container_width=True)
        # 图表展示
        st.write("### 📈 Nutrition Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Calories", f"{nutrition['energy']} kcal", "🔥")
            st.metric("Calories per Serving", f"{round(nutrition['energy']/4)} kcal", "🍽️")
        with col2:
            st.metric("Protein", f"{nutrition['protein']}g", "{:.1f}% of daily value".format(nutrition['protein']/50*100))
            st.metric("Carbs", f"{nutrition['carbohydrates']}g", "{:.1f}% of daily value".format(nutrition['carbohydrates']/275*100))
        with col3:
            st.metric("Fat", f"{nutrition['fat']}g", "{:.1f}% of daily value".format(nutrition['fat']/65*100))
            st.metric("Fiber", "5g", "20% of daily value")
        # 饼图
        st.write("### 🥧 Macronutrient Distribution (Pie Chart)")
        import pandas as pd
        pie_data = pd.DataFrame({
            "Macronutrient": ["Protein", "Fat", "Carbohydrates"],
            "Value": [nutrition["protein"]*4, nutrition["fat"]*9, nutrition["carbohydrates"]*4]
        })
        st.plotly_chart({
            "data": [{
                "labels": pie_data["Macronutrient"],
                "values": pie_data["Value"],
                "type": "pie",
                "hole": .3
            }],
            "layout": {"title": "Macronutrient Energy Distribution"}
        }, use_container_width=True)
        # 雷达图
        st.write("### 🕸️ Nutrition Radar Chart")
        radar_data = pd.DataFrame({
            "Nutrient": ["Calories", "Protein", "Fat", "Carbs"],
            "Value": [nutrition["energy"], nutrition["protein"]*4, nutrition["fat"]*9, nutrition["carbohydrates"]*4]
        })
        st.plotly_chart({
            "data": [{
                "type": "scatterpolar",
                "r": radar_data["Value"],
                "theta": radar_data["Nutrient"],
                "fill": "toself",
                "name": "Nutrition"
            }],
            "layout": {"polar": {"radialaxis": {"visible": True}}, "showlegend": False, "title": "Nutrition Radar"}
        }, use_container_width=True)
        # 条形图
        st.write("### 📊 Ingredient Nutrition Comparison (Bar Chart)")
        if nutrition_table:
            df_bar = pd.DataFrame(nutrition_table)
            st.bar_chart(df_bar.set_index("Ingredient")[ ["Calories", "Protein (g)", "Fat (g)", "Carbs (g)"] ])
        # 健康评分
        health_score = min(100, max(0, 
            70 + 
            (10 if nutrition['protein'] >= 20 else -5) +
            (-5 if nutrition['fat'] > 15 else 5) +
            (5 if nutrition['carbohydrates'] < 40 else -5)
        ))
        st.write(f"### 🏅 Health Score: <span style='color:#43A047;font-size:22px;'><b>{health_score}/100</b></span>", unsafe_allow_html=True)
        st.progress(health_score/100)
        # 文字分析
        st.subheader("📝 Nutrition Insights")
        analysis_text = []
        if nutrition["energy"] < 300:
            analysis_text.append("🍃 Low calorie meal suitable for weight management")
        elif 300 <= nutrition["energy"] <= 500:
            analysis_text.append("⚖️ Moderate calorie content for balanced nutrition")
        else:
            analysis_text.append("🔥 High energy meal - great for active lifestyles")
        if nutrition["protein"] >= 20:
            analysis_text.append("💪 High protein content supports muscle health")
        elif nutrition["protein"] < 10:
            analysis_text.append("🥩 Consider adding more protein sources")
        if nutrition["fat"] > 15:
            analysis_text.append("🧈 High fat content - pair with fiber-rich foods")
        elif nutrition["fat"] < 5:
            analysis_text.append("❤️ Low fat content promotes heart health")
        if nutrition["carbohydrates"] > 40:
            analysis_text.append("🍚 High carb meal provides sustained energy")
        else:
            analysis_text.append("🌾 Balanced carbohydrate content")
        st.write("This meal provides:")
        for item in analysis_text:
            st.write(f"- {item}")
        st.write("\n**💡 Chef's Tips:**")
        tips_generated = False
        if "High fat" in analysis_text and "High calorie" in analysis_text:
            st.write("- Consider adding fresh vegetables to balance the meal 🥗")
            tips_generated = True
        if "Low calorie" in analysis_text:
            st.write("- Pair with a protein shake for a complete nutritional profile 🥤")
            tips_generated = True
        if not tips_generated:
            st.write("- Season generously for better flavor! 🧂")
            st.write("- Don't overcrowd the pan when cooking for even results. 🔥")
            st.write("- Taste and adjust seasonings throughout the cooking process. 😋")
        st.markdown("<hr style='border: 2px solid #43A047;'>", unsafe_allow_html=True)
        st.balloons()

if __name__ == "__main__":
    main()