# 真正有意义的营养导向推荐系统
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM




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



def get_nutrition_data(ingredients):
    total = {"energy": 0, "fat": 0, "carbohydrates": 0, "protein": 0}
    for ingredient in ingredients:
        if ingredient in nutrition_db:
            data = nutrition_db[ingredient]
            total["energy"] += data["energy"]
            total["fat"] += data["fat"]
            total["carbohydrates"] += data["carbohydrates"]
            total["protein"] += data["protein"]
        
    return total
    
    
    
MODEL_PATH = "nutrition-recipe-model/"
    
def generate_recipe_by_nutrition_goal(ingredients_list, nutrition_goal):

    ingredients_text = ", ".join(ingredients_list)
    
    prompt = f"Recommend a {nutrition_goal} recipe using some or all of these ingredients: {ingredients_text}"
    
    tokenizer = AutoTokenizer.from_pretrained("model/")
    model = AutoModelForSeq2SeqLM.from_pretrained("model/")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text
    
    return response

test_ingredients = ["chicken breast", "mushroom", "broccoli", "garlic",
                    "ginger", "basil", "olive oil", "salt", "pepper", "soy sauce",
                    "brown rice"]


print("-"*100)
print("Test ingredients: ", test_ingredients)

print("-"*100)

print("Low-calorie recipe recommendation:")
print(generate_recipe_by_nutrition_goal(test_ingredients, "low-calorie"))

print("-"*100)

print("\nHigh-protein recipe recommendation:")
print(generate_recipe_by_nutrition_goal(test_ingredients, "high-protein"))

print("-"*100)
