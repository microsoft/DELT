from datasets import load_dataset


dataset_name = "GAIR/lima"  
save_path = "pretrain_data/lima"  

dataset = load_dataset(dataset_name)

dataset.save_to_disk(save_path)

print(f"Dataset '{dataset_name}' has been saved to '{save_path}'")