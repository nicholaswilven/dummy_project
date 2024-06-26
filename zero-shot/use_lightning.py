from lightning import LightningModule, LightningDataModule, Trainer
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader, random_split, default_collate
from torch import nn, optim
from lightning.pytorch.loggers import WandbLogger
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import os

from huggingface_hub import login
login(os.getenv("ACCESS_TOKEN"))

WEIGHT_DECAY=0.01
LEARNING_RATE=1e-5
EPOCH=2
BATCH_SIZE=128
VAL_SIZE=0.05
block_size = 256
HUB_MODEL_NAME="awidjaja/zero-shot-xlmR-food"
DATASET_NAME="awidjaja/compiled_nli"
ACCELERATOR="tpu"
BASE_MODEL_NAME="awidjaja/pretrained-xlmR-food"
NUM_WORKERS = 64

def collator(batch):
    keys = batch[0].keys()
    new_batch = {}
    for key in keys:
        if key == "labels":
            labels = torch.cat([(example[key].view(1, -1)) for example in batch], dim = 0).view(-1)
        else:
            new_batch[key] = torch.cat([(example[key].view(1, -1)) for example in batch], dim = 0)
    return new_batch, labels

def tokenize(dataset, tokenizer):
    def batch_tokenize(batch):
        token = tokenizer(
            batch["premise"],
            batch["hypothesis"],
            max_length = block_size,
            padding = "max_length",
            truncation = "only_first",
            return_tensors = "pt",
        )
        return token

    return dataset.map(
        batch_tokenize,
        num_proc = NUM_WORKERS,
        batched = True,
        remove_columns = ["hypothesis", "premise"],
    )

class Model(LightningModule):
    def __init__(self, model_name: str = "xlm-roberta-base", label_index: dict = {}):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels = len(label_index),
            label2id = label_index,
            id2label = {v:k for k,v in label_index.items()}
        )
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

        # Remove LayerNorm from weight decay params
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": WEIGHT_DECAY,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
        return optimizer


class NLIData(LightningDataModule):
    def __init__(self, model_name: str = "xlm-roberta-base", label_index = {}):
        super().__init__()
        dataset = load_dataset(DATASET_NAME, split = "train")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dataset = tokenize(dataset, self.tokenizer)
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        self.dataset = dataset.train_test_split(test_size=VAL_SIZE, seed = 42)
        self.train_dataset = self.dataset['train'].shuffle(seed = 42)
        self.val_dataset = self.dataset['test'].shuffle(seed = 42)
    
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle = True, batch_size = BATCH_SIZE, collate_fn = collator, num_workers = NUM_WORKERS)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = BATCH_SIZE, collate_fn = collator, num_workers = NUM_WORKERS)

if __name__ == "__main__":
    label_index = {
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2
        }
    model = Model(model_name = BASE_MODEL_NAME, label_index = label_index)
    data = NLIData(model_name = BASE_MODEL_NAME, label_index = label_index)
    checkpoint_callback = ModelCheckpoint(monitor = 'val_loss')
    wandblogger = WandbLogger(
        log_model = True,
        mode = "online",
        project = "madral-aspect-labelling",
        config = {
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCH,
            "batch_size": BATCH_SIZE,
            "block_size": block_size,
            "base_model_name": BASE_MODEL_NAME,
            "hub_model_name": HUB_MODEL_NAME,
            "dataset_name": DATASET_NAME
            }
        )
    wandblogger.experiment
    trainer = Trainer(
        accelerator = ACCELERATOR,
        devices = "auto",
        max_epochs = EPOCH,
        callbacks =  [checkpoint_callback],
        logger = wandblogger
        )
    trainer.fit(model, data)
    wandb.finish()
    
    model.model.push_to_hub(HUB_MODEL_NAME, private = True)
    data.tokenizer.push_to_hub(HUB_MODEL_NAME, private = True)