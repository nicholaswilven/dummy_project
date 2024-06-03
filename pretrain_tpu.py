from lightning import LightningModule, LightningDataModule, Trainer
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from transformers.modeling_outputs import MaskedLMOutput
from torch.utils.data import DataLoader
from torch import nn, optim
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from typing import List, Optional, Tuple, Union
import os
import torch_xla.core.xla_model as xm
import gc

from huggingface_hub import login
login(os.getenv("ACCESS_TOKEN"))

WEIGHT_DECAY=0.01
LEARNING_RATE=5e-5
MLM_PROB=0.15
VAL_SIZE=0.05
EPOCH=4
DUPLICATE=7
BATCH_SIZE=8
HUB_MODEL_NAME="awidjaja/pretrained-xlmR-food"
ACCELERATOR="tpu"
BASE_MODEL_NAME="xlm-roberta-base"
NUM_WORKERS=32

def tokenize(dataset, tokenizer):
    def batch_tokenize(batch):
        return tokenizer(
            batch["text"],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    return dataset.map(
        batch_tokenize,
        num_proc=512,
        batched=True,
        remove_columns=["text"],
    )

class FoodModel(LightningModule):
    def __init__(self, model_name: str = "xlm-roberta-base", total_steps = 0, warmup_steps = 0):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.save_hyperparameters()
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss

        # Calculate accuracy
        predictions = torch.argmax(outputs.logits, dim=-1)
        labels = batch['labels']
        mask = labels != -100
        correct = (predictions[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        accuracy = correct / total if total > 0 else 0

        self.log("train_accuracy", accuracy, on_epoch=True, on_step=True)
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss

        # Calculate accuracy
        predictions = torch.argmax(outputs.logits, dim=-1)
        labels = batch['labels']
        mask = labels != -100
        correct = (predictions[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        accuracy = correct / total if total > 0 else 0

        self.log("val_accuracy", accuracy, on_epoch=True, on_step=True)
        self.log("val_loss", loss, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": WEIGHT_DECAY,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
        # Define warmup scheduler
        def warmup_scheduler(step):
            return min(1.0, step / self.warmup_steps)
        
        # Define linear decay scheduler
        def linear_decay_scheduler(step):
            return max(0.0, 1.0 - step / self.total_steps)
        
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: warmup_scheduler(step) * linear_decay_scheduler(step))
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

class FoodDataModule(LightningDataModule):
    def __init__(self, model_name: str = "xlm-roberta-base"):
        super().__init__()
        concat_ds = load_dataset("awidjaja/512-food-dataset", split = "train")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dataset = tokenize(concat_ds, self.tokenizer)
        self.dataset.set_format("torch", columns = ["input_ids", "attention_mask"])
        self.dataset = self.dataset.train_test_split(test_size=VAL_SIZE, seed = 42)
        self.train_dataset = self.dataset['train'].shuffle(seed = 42)
        self.val_dataset = self.dataset['test'].shuffle(seed = 42)
        self.train_dataset = concatenate_datasets([self.train_dataset]*DUPLICATE)
        self.val_dataset = concatenate_datasets([self.val_dataset]*DUPLICATE)
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=MLM_PROB)
    
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle = True, collate_fn=self.data_collator, num_workers=NUM_WORKERS)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=BATCH_SIZE,  shuffle = False, collate_fn=self.data_collator, num_workers=NUM_WORKERS)

def main():
    data = FoodDataModule(model_name=BASE_MODEL_NAME)
    TOTAL_STEPS = EPOCH*len(data.train_dataloader())
    WARMUP_STEPS = int(0.1*TOTAL_STEPS)
    model = FoodModel(model_name = BASE_MODEL_NAME, total_steps = TOTAL_STEPS, warmup_steps = WARMUP_STEPS)
    checkpoint_callback = ModelCheckpoint(
        monitor = 'val_loss',
        save_top_k = 4
    )
    trainer = Trainer(
        accelerator = ACCELERATOR,
        max_epochs = EPOCH,
        callbacks = [checkpoint_callback],
        accumulate_grad_batches = 32,
        precision = '16-true'
    )
    trainer.fit(model, data)
    print("Best Model Checkpoint:", checkpoint_callback.best_model_path)
    model.model.push_to_hub(HUB_MODEL_NAME, use_auth_token = os.getenv("ACCESS_TOKEN"), private = True)
    data.tokenizer.push_to_hub(HUB_MODEL_NAME, use_auth_token = os.getenv("ACCESS_TOKEN"), private = True)

if __name__ == "__main__":
    main()
