from lightning import LightningModule, LightningDataModule, Trainer
from datasets import load_dataset, concatenate_datasets
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, random_split, default_collate
from torch import nn, optim
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from lightning.pytorch.loggers import WandbLogger
import wandb

import os
import dotenv
if '.env' in os.listdir():
    dotenv.load_dotenv('~/sentiment/.env')

from huggingface_hub import login
login(os.getenv("ACCESS_TOKEN"))

NUM_WORKERS = 64
BATCH_SIZE=128
EPOCH=3
WEIGHT_DECAY=0.01
VAL_SIZE=0.05
LEARNING_RATE=0.00001
block_size = 256
BASE_MODEL_NAME="awidjaja/pretrained-xlmR-food"
EN_DATASET="carant-ai/english_sentiment_dataset"
ID_DATASET="carant-ai/indonesian_sentiment_dataset"
PJRT_DEVICE="TPU"
ACCELERATOR="tpu"
HUB_MODEL_NAME="awidjaja/sentiment-xlmR-base"

def collator(batch):
    keys = batch[0].keys()
    new_batch = {}
    for key in keys:
        if key == "labels":
            labels = torch.cat([(example[key].view(1, -1)) for example in batch], dim = 0).view(-1)
        else:
            new_batch[key] = torch.cat([(example[key].view(1, -1)) for example in batch], dim = 0)
    return new_batch, labels

def tokenize(dataset, tokenizer, label_index):
    def batch_tokenize(batch):
        token = tokenizer(
            batch["text"],
            max_length = block_size,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt",
        )

        token["labels"] = [label_index[label] for label in batch["label_text"]]
        return token

    return dataset.map(
        batch_tokenize,
        num_proc = NUM_WORKERS,
        batched = True,
        remove_columns = ["text", "label_text", "source", "split"],
    )

class Model(LightningModule):
    def __init__(self, model_name: str = "xlm-roberta-base", label_index: dict = {} , total_steps = 0, warmup_steps = 0):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels = len(label_index),
            label2id = label_index,
            id2label = {v:k for k,v in label_index.items()}
        )
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.save_hyperparameters()
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        features, labels = batch
        logits = self.model(**features).logits
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, on_epoch = True, on_step = True)
        predicted_labels = logits.argmax(axis=1).view(-1)
        accuracy = (predicted_labels == labels).float().mean()
        self.log("train_accuracy", accuracy, on_epoch = True, on_step = True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        logits = self.model(**features).logits
        loss = self.loss_fn(logits, labels)
        self.log("val_loss", loss, on_epoch = True, on_step = True)
        predicted_labels = logits.argmax(axis=1).view(-1)
        accuracy = (predicted_labels == labels).float().mean()
        self.log("val_accuracy", accuracy, on_epoch = True, on_step = True)
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


class Sentence(LightningDataModule):
    def __init__(self, model_name: str = "xlm-roberta-base", val_ratio = 0.05, label_index = {}):
        super().__init__()
        id_dataset = load_dataset(
            ID_DATASET, split = "train",
        )
        en_dataset = load_dataset(
            EN_DATASET, split = "train",
        )
        concat_ds = concatenate_datasets([id_dataset, en_dataset]).shuffle(seed=42)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dataset = tokenize(concat_ds, self.tokenizer, label_index)
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        dataset = dataset.train_test_split(test_size = val_ratio)
        self.train_dataset = dataset['train']
        self.val_dataset = dataset['test']
    
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = BATCH_SIZE, collate_fn = collator, num_workers = NUM_WORKERS)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = BATCH_SIZE, collate_fn = collator, num_workers = NUM_WORKERS)

def main():
    label_index = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
        }
    data = Sentence(model_name = BASE_MODEL_NAME, val_ratio = VAL_SIZE, label_index = label_index)
    TOTAL_STEPS = EPOCH*len(data.train_dataloader())
    WARMUP_STEPS = int(0.1*TOTAL_STEPS)
    model = Model(model_name = BASE_MODEL_NAME, label_index = label_index, total_steps = TOTAL_STEPS, warmup_steps = WARMUP_STEPS)
    checkpoint_callback = ModelCheckpoint(monitor = 'val_loss')
    wandblogger = WandbLogger(
        log_model = True,
        mode = "online",
        project = "intial-sentiment",
        config = {
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCH,
            "batch_size": BATCH_SIZE,
            "block_size": block_size,
            "base_model_name": BASE_MODEL_NAME,
            "hub_model_name": HUB_MODEL_NAME,
            "dataset_name": ID_DATASET +","+EN_DATASET
            }
        )
    wandblogger.experiment
    trainer = Trainer(
        accelerator = ACCELERATOR,
        max_epochs = EPOCH,
        callbacks =  [checkpoint_callback],
        logger = wandblogger
        )
    trainer.fit(model, data)
    
    model.model.push_to_hub(HUB_MODEL_NAME, private = True)
    data.tokenizer.push_to_hub(HUB_MODEL_NAME, private = True)

if __name__ == "__main__":
    main()