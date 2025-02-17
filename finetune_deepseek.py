from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from torch.utils.data import Dataset, random_split
import torch
import json
import os

# Custom dataset class for JSON files
class CustomDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load all JSON files in the directory
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.json'):
                with open(os.path.join(data_dir, file_name), 'r') as f:
                    file_data = json.load(f)
                    self.data.extend(file_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        completion = item['completion']
        
        # Format with special tokens
        text = f"<|startoftext|>User: {prompt}\nAssistant: {completion}<|endoftext|>"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

def main():
    # Model setup
    model_id = "deepseek-ai/deepseek-llm-7b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if torch.backends.mps.is_available():
        print("MPS backend is available!")
    else:
        print("MPS backend is not available.")

    # Dataset preparation
    dataset = CustomDataset(
        data_dir='TrainingData',
        tokenizer=tokenizer,
        max_length=2048
    )
    
    # Train/validation split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Training configuration
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=100,
        weight_decay=0.05,
        logging_dir="./logs",
        logging_steps=5,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        fp16=True,
        gradient_accumulation_steps=16,
        learning_rate=1e-5,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        save_total_limit=3,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model.to(device),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda data: {
            'input_ids': torch.stack([item['input_ids'] for item in data]).to(device),
            'attention_mask': torch.stack([item['attention_mask'] for item in data]).to(device),
            'labels': torch.stack([item['labels'] for item in data]).to(device)
        }
    )

    # Start training
    trainer.train()
    model.save_pretrained("./fine_tuned_deepseek")
    tokenizer.save_pretrained("./fine_tuned_deepseek")

if __name__ == "__main__":
    main() 