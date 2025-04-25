streamlit run app.py

```js
import openfoodfacts
# Load model directly
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation



# 设置随机种子保证可重复性
torch.manual_seed(42)

# 加载CLIP模型
#device = "cuda" if torch.cuda.is_available() else "cpu"
#model, preprocess = clip.load("ViT-B/32", device=device)
processor = AutoImageProcessor.from_pretrained("priteshkeleven/FoodSeg103-mit-b0-fine-tuned")
model = SegformerForSemanticSegmentation.from_pretrained("priteshkeleven/FoodSeg103-mit-b0-fine-tuned")
# 定义食材候选列表（可根据需要扩展）
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
```


![image](https://github.com/user-attachments/assets/527199ac-8029-4aef-a8e4-33c85141df31)


![image](https://github.com/user-attachments/assets/eccb173f-1b6e-4a5a-b25a-233a57acb996)


![image](https://github.com/user-attachments/assets/e2507647-a907-420c-a28d-11ce21df3198)

![image](https://github.com/user-attachments/assets/2aa949f8-2db7-4a38-b75f-97d26602a489)


![image](https://github.com/user-attachments/assets/863cbdfa-21c1-4dfa-9969-e01eb54413dd)


mask2form ram崩溃

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
Epoch 1/50 - Training: 100%|██████████| 54/54 [00:32<00:00,  1.65it/s]
Epoch 1 Training Loss: 0.2382
Validation: 100%|██████████| 14/14 [00:04<00:00,  3.24it/s]
Epoch 1 Validation Loss: 0.1214

Epoch 2/50 - Training: 100%|██████████| 54/54 [00:22<00:00,  2.36it/s]
Epoch 2 Training Loss: 0.1070
Validation: 100%|██████████| 14/14 [00:03<00:00,  3.68it/s]
Epoch 2 Validation Loss: 0.1078

Epoch 3/50 - Training: 100%|██████████| 54/54 [00:22<00:00,  2.38it/s]
Epoch 3 Training Loss: 0.0880
Validation: 100%|██████████| 14/14 [00:03<00:00,  3.59it/s]
Epoch 3 Validation Loss: 0.1023

Epoch 4/50 - Training: 100%|██████████| 54/54 [00:23<00:00,  2.25it/s]
Epoch 4 Training Loss: 0.0722
Validation: 100%|██████████| 14/14 [00:03<00:00,  3.66it/s]
Epoch 4 Validation Loss: 0.0995

Epoch 5/50 - Training: 100%|██████████| 54/54 [00:22<00:00,  2.40it/s]
Epoch 5 Training Loss: 0.0592
Validation: 100%|██████████| 14/14 [00:03<00:00,  3.56it/s]
Epoch 5 Validation Loss: 0.0982

Epoch 6/50 - Training: 100%|██████████| 54/54 [00:22<00:00,  2.39it/s]
Epoch 6 Training Loss: 0.0490
Validation: 100%|██████████| 14/14 [00:03<00:00,  3.59it/s]
Epoch 6 Validation Loss: 0.0989

Validation loss did not improve. Counter: 1/5
Epoch 7/50 - Training: 100%|██████████| 54/54 [00:22<00:00,  2.38it/s]
Epoch 7 Training Loss: 0.0412
Validation: 100%|██████████| 14/14 [00:03<00:00,  3.67it/s]
Epoch 7 Validation Loss: 0.0992

Validation loss did not improve. Counter: 2/5
Epoch 8/50 - Training: 100%|██████████| 54/54 [00:23<00:00,  2.34it/s]
Epoch 8 Training Loss: 0.0348
Validation: 100%|██████████| 14/14 [00:04<00:00,  3.48it/s]
Epoch 8 Validation Loss: 0.1001

Validation loss did not improve. Counter: 3/5
Epoch 9/50 - Training: 100%|██████████| 54/54 [00:22<00:00,  2.39it/s]
Epoch 9 Training Loss: 0.0296
Validation: 100%|██████████| 14/14 [00:04<00:00,  3.01it/s]
Epoch 9 Validation Loss: 0.0998

Validation loss did not improve. Counter: 4/5
Epoch 10/50 - Training: 100%|██████████| 54/54 [00:22<00:00,  2.43it/s]
Epoch 10 Training Loss: 0.0248
Validation: 100%|██████████| 14/14 [00:04<00:00,  2.84it/s]Epoch 10 Validation Loss: 0.1017

Validation loss did not improve. Counter: 5/5
Early stopping triggered at epoch 10!
Training completed. Best validation loss: 0.0982
```
