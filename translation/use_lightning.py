from lightning import LightningModule, LightningDataModule, Trainer
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
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

WEIGHT_DECAY=0.001
LEARNING_RATE=1e-5
VAL_SIZE=0.05
EPOCH=1
BATCH_SIZE=8

block_size = 512
HUB_MODEL_NAME="thonyyy/komodo-7b-translate-p1"
DATASET_NAME="thonyyy/tatoeba-nusax-mt-p2"
ACCELERATOR="tpu"
BASE_MODEL_NAME="Yellow-AI-NLP/komodo-7b-base"
NUM_WORKERS = 64
text_column_name = 'text'

def tokenize(dataset, tokenizer):
    def prompting_template_and_tokenize(examples):
        # Concatenate all texts.
        # columns : ['source_lang','target_lang','source_text','target_text']
        prompt_list = []
        for source_lang, target_lang, source_text, target_text in zip(examples['source_lang'],examples['target_lang'],examples['source_text'],examples['target_text']):
            source_text = tokenizer.decode(tokenizer(source_text, add_special_tokens = False).input_ids[:(block_size-100)//2])
            target_text = tokenizer.decode(tokenizer(target_text, add_special_tokens = False).input_ids[:(block_size-100)//2])
            prompt = f"""Di bawah ini adalah instruksi yang menjelaskan tugas, dipasangkan dengan masukan yang memberikan konteks lebih lanjut. Tulis respons yang secara tepat melengkapi permintaan.

### Instruksi:
Terjemahkan teks berikut dari bahasa {source_lang.replace('_',' ').title()} ke bahasa {target_lang.replace('_',' ').title()}.

### Masukan:
{source_text}

### Respon:
{target_text} </s>"""
            prompt_list.append(prompt)
        outputs = tokenizer(
            prompt_list,
            max_length=block_size,
            padding="max_length",
            truncation=True
        ) 
        
        outputs['labels'] = [x.copy() for x in outputs['input_ids']]
        return outputs
    return dataset.map(
        prompting_template_and_tokenize,
        num_proc=NUM_WORKERS,
        batched=True,
        # remove_columns=["text"],
    )

class LargeLanguageModel(LightningModule):
    def __init__(self, model_name: str = "", total_steps = 0, warmup_steps = 0):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.save_hyperparameters()
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Apply LoRA
        self.apply_lora()

    def apply_lora(self):
        from peft import get_peft_model, LoraConfig, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        self.model = get_peft_model(self.model, lora_config)
        
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss

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

class LargeDataModule(LightningDataModule):
    def __init__(self, model_name: str = ""):
        super().__init__()
        concat_ds = load_dataset(DATASET_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        self.dataset = tokenize(concat_ds, self.tokenizer)
        self.dataset.set_format("torch", columns = ["input_ids", "attention_mask", "labels"])
        self.dataset = self.dataset.train_test_split(test_size=VAL_SIZE, seed = 42)
        self.train_dataset = self.dataset['train'].shuffle(seed = 42)
        self.val_dataset = self.dataset['test'].shuffle(seed = 42)
        
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle = True, collate_fn=self.data_collator, num_workers=NUM_WORKERS)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=BATCH_SIZE,  shuffle = False, collate_fn=self.data_collator, num_workers=NUM_WORKERS)

def main():
    data = LargeDataModule(model_name=BASE_MODEL_NAME)
    TOTAL_STEPS = 1000
    WARMUP_STEPS = int(0.1*TOTAL_STEPS)
    model = LargeLanguageModel(model_name = BASE_MODEL_NAME, total_steps = TOTAL_STEPS, warmup_steps = WARMUP_STEPS)
    checkpoint_callback = ModelCheckpoint(
        monitor = 'val_loss',
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
