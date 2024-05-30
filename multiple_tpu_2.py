import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, DistributedSampler
from lightning import LightningModule, LightningDataModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from datasets import load_dataset, concatenate_datasets
from torch.optim.lr_scheduler import LambdaLR
import os
import torch
from typing import List, Optional, Tuple, Union

WEIGHT_DECAY = 0.01
LEARNING_RATE = 5e-5
MLM_PROB = 0.15
VAL_SIZE = 0.05
EPOCH = 4
DUPLICATE = 5
BATCH_SIZE = 8
HUB_MODEL_NAME = "awidjaja/pretrained-xlmR-food"
BASE_MODEL_NAME = "xlm-roberta-base"
NUM_WORKERS = 32

class FoodModel(LightningModule):
    def __init__(self, model_name: str = "xlm-roberta-base", total_steps=0, warmup_steps=0):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.save_hyperparameters()
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("val_loss", loss, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
        
        def warmup_scheduler(step):
            return min(1.0, step / self.warmup_steps)
        
        def linear_decay_scheduler(step):
            return max(0.0, 1.0 - step / self.total_steps)
        
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: warmup_scheduler(step) * linear_decay_scheduler(step))
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

class FoodDataModule(LightningDataModule):
    def __init__(self, model_name: str = "xlm-roberta-base"):
        super().__init__()
        concat_ds = load_dataset("awidjaja/512-food-dataset", split="train", token=os.getenv("ACCESS_TOKEN")).take(2048)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dataset = self.tokenize(concat_ds)
        self.dataset.set_format("torch", columns=["input_ids", "attention_mask"])
        self.dataset = self.dataset.train_test_split(test_size=VAL_SIZE)
        self.train_dataset = self.dataset['train']
        self.val_dataset = self.dataset['test']
        self.train_dataset = concatenate_datasets([self.train_dataset] * DUPLICATE)
        self.val_dataset = concatenate_datasets([self.val_dataset] * DUPLICATE)
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=MLM_PROB)

    def tokenize(self, dataset):
        def batch_tokenize(batch):
            return self.tokenizer(
                batch["text"],
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        return dataset.map(
            batch_tokenize,
            batched=True,
            remove_columns=["text"],
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=self.data_collator,
            num_workers=NUM_WORKERS,
            shuffle=True
        )
        return pl.MpDeviceLoader(train_loader, xm.xla_device())

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=self.data_collator,
            num_workers=NUM_WORKERS,
            shuffle=False
        )
        return pl.MpDeviceLoader(val_loader, xm.xla_device())

def _mp_fn(index, flags):
    data = FoodDataModule(model_name=BASE_MODEL_NAME)
    data.setup()
    TOTAL_STEPS = EPOCH * len(data.train_dataloader())
    WARMUP_STEPS = int(0.1 * TOTAL_STEPS)
    model = FoodModel(model_name=BASE_MODEL_NAME, total_steps=TOTAL_STEPS, warmup_steps=WARMUP_STEPS)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')
    trainer = Trainer(
        accelerator='xla',
        max_epochs=EPOCH,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=32,
        precision=16
    )
    trainer.fit(model, data)
    xm.rendezvous("download_only_once")
    if xm.is_master_ordinal():
        model = FoodModel.load_from_checkpoint(checkpoint_callback.best_model_path)
        model.model.push_to_hub(HUB_MODEL_NAME)
        data.tokenizer.push_to_hub(HUB_MODEL_NAME)

if __name__ == "__main__":
    FLAGS = {}
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')
