import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from trl import SFTTrainer
import random


MAX_SAMPLES = 1000

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

# Get nutrition information for ingredients
def get_nutrition_data(ingredients):
    total = {"energy": 0, "fat": 0, "carbohydrates": 0, "protein": 0}
    count = 0
    
    for ingredient in ingredients:
        # Try to find the ingredient in the database
        for key in nutrition_db:
            if key in ingredient.lower():
                data = nutrition_db[key]
                total["energy"] += data["energy"]
                total["fat"] += data["fat"]
                total["carbohydrates"] += data["carbohydrates"]
                total["protein"] += data["protein"]
                count += 1
                break
    
    # If no ingredients were found, provide default values
    if count == 0:
        return {"calories": 200, "protein": 10, "carbs": 25, "fat": 8}
    
    # Convert to the format used in the training function
    return {
        "calories": total["energy"],
        "protein": total["protein"],
        "carbs": total["carbohydrates"],
        "fat": total["fat"]
    }

def filter_and_prepare_targeted_dataset(dataset):
    
    # 初始化计数器和结果列表
    low_calorie_count = 0
    high_protein_count = 0
    filtered_examples = []
    
    # 确定食谱中的食材是否在我们的营养数据库中
    def has_known_ingredients(ingredients_list):
        for ingredient in ingredients_list:
            ingredient_lower = ingredient.lower()
            for known_ingredient in nutrition_db:
                if known_ingredient in ingredient_lower:
                    return True
        return False
    
    # 遍历数据集
    for i, example in enumerate(dataset):
        # 如果两种策略都已收集足够的样本，停止处理
        if low_calorie_count >= MAX_SAMPLES and high_protein_count >= MAX_SAMPLES:
            break
            
        ingredients = example["ingredients"]
        
        # 检查该食谱是否包含我们已知营养数据的食材
        if not has_known_ingredients(ingredients):
            continue
            
        # 获取营养信息
        nutrition = get_nutrition_data(ingredients)
        
        # 检查是否符合低卡标准且还需要低卡样本
        is_low_calorie = nutrition['calories'] <= 300 and low_calorie_count < MAX_SAMPLES
        # 检查是否符合高蛋白标准且还需要高蛋白样本
        is_high_protein = nutrition['protein'] >= 20 and high_protein_count < MAX_SAMPLES
        
        # 如果两个条件都不满足，跳过此食谱
        if not (is_low_calorie or is_high_protein):
            continue
            
        # 如果同时满足两个条件，选择数量较少的类别
        if is_low_calorie and is_high_protein:
            if low_calorie_count <= high_protein_count:
                nutrition_goal = "low-calorie"
                low_calorie_count += 1
            else:
                nutrition_goal = "high-protein"
                high_protein_count += 1
        elif is_low_calorie:
            nutrition_goal = "low-calorie"
            low_calorie_count += 1
        else:  # is_high_protein
            nutrition_goal = "high-protein"
            high_protein_count += 1
        
        # 格式化训练样本
        title = example["title"]
        directions = example["directions"]
        
        prompt = f"Recommend a {nutrition_goal} recipe using some or all of these ingredients: {', '.join(ingredients)}"
        response = (f"Recipe name: {title}\n\n"
                   f"Ingredients & quantities:\n{', '.join(ingredients)}\n\n"
                   f"Instruction:\n{'; '.join(directions)}\n\n")
        
        filtered_examples.append({"text": f"{prompt}\n\n{response}"})
        
        # 打印进度
        if (i+1) % 10000 == 0:
            print(f"已处理 {i+1} 条食谱，找到低卡食谱 {low_calorie_count} 条，高蛋白食谱 {high_protein_count} 条")
    
    print(f"最终数据集统计：低卡食谱 {low_calorie_count} 条，高蛋白食谱 {high_protein_count} 条")
    return Dataset.from_list(filtered_examples)

def train_nutrition_lora():
    dataset = load_dataset("mbien/recipe_nlg", data_dir="./data")
    print(f"原始数据集加载成功。结构: {dataset}")
    
    processed_dataset = filter_and_prepare_targeted_dataset(dataset["train"])
    print(f"筛选后的数据集大小: {len(processed_dataset)}")
    
    # Shuffle the dataset
    processed_dataset = processed_dataset.shuffle(seed=42)
    
    print(f"Using {len(processed_dataset)} examples for nutrition fine-tuning")
    
    # Set up model, device
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print(f"Using device: {device}")
    
    # Model configuration
    MODEL_NAME = "model/"  # model path from part2.
    
    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, pad_token='<|endoftext|>')
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Prepare model for kbit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,                    # LoRA rank
        lora_alpha=32,           # LoRA scaling factor
        lora_dropout=0.05,       # LoRA dropout rate
        bias="none",             
        task_type="CAUSAL_LM",   
        target_modules="all-linear"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.to(device)
    print(model.print_trainable_parameters())
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./qwen-nutrition-recipe",
        num_train_epochs=3,
        per_device_train_batch_size=1,   # Small batch size for Windows
        gradient_accumulation_steps=4,   # Accumulate gradients
        optim="adamw_torch",             # Use native torch optimizer
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",    # Constant learning rate
        report_to="tensorboard",
        dataloader_num_workers=0         # Better for Windows
    )
    
    # Create SFT trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=processed_dataset,
        args=training_args,
        tokenizer=tokenizer
    )
    
    # Train the model
    trainer.train()
    
    # Save the trained model
    model_save_dir = "./nutrition-recipe-model"
    trainer.model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    print(f"Nutrition-oriented recipe model saved to: {model_save_dir}")

if __name__ == "__main__":
    train_nutrition_lora()