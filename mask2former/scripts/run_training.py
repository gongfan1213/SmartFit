import yaml
from scripts.train import SegmentationTrainer
from datasets import load_dataset
from huggingface_hub import login, hf_hub_download
import json
import os


def load_config(config_file):
    """
    Load configuration settings from a YAML file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration parameters.
    """
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    # Load configuration from YAML file
    config = load_config("config.yaml")

    # Specify the local dataset path and the filename for id2label mapping
    local_dataset_path = r"C:\Users\宫凡\.cache\huggingface\datasets\FoodSeg103" # Use raw string for Windows path
    filename = "id2label.json"

    # Load id2label mapping from the local path
    id2label_path = os.path.join(local_dataset_path, filename)
    with open(id2label_path, "r") as file:
        id2label = json.load(file)

    # Convert keys to integers
    id2label = {int(k): v for k, v in id2label.items()}
    print(id2label)

    # Load training and validation datasets from the local path
    train_dataset = load_dataset("EduardoPacheco/FoodSeg103", split="train", data_dir=local_dataset_path)
    val_dataset = load_dataset("EduardoPacheco/FoodSeg103", split="validation", data_dir=local_dataset_path)

    # Initialize the SegmentationTrainer with loaded configuration and datasets
    trainer = SegmentationTrainer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        id2label=id2label,
        batch_size=config["batch_size"],
        lr=config["learning_rate"],
        epochs=config["epochs"],
        save_path=config["save_path"],
        load_checkpoint=config["load_checkpoint"],
        log_dir=config["log_dir"],
    )

    # Start training the model
    trainer.train()
