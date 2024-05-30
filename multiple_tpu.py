from torch.utils.data.distributed import DistributedSampler
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
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

WEIGHT_DECAY=0.01
LEARNING_RATE=5e-5
MLM_PROB=0.15
VAL_SIZE=0.05
EPOCH=4
DUPLICATE=5
BATCH_SIZE=64
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
        num_proc=NUM_WORKERS,
        batched=True,
        remove_columns=["text"],
    )

class FoodModel(LightningModule):
    def __init__(self, model_name: str = "xlm-roberta-base", total_steps=0, warmup_steps=0):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.save_hyperparameters()
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.loss_fn = nn.CrossEntropyLoss()

        def lm_head_forward(self, features, labels, hidden_size, **kwargs):
            x = self.dense(features)
            x = torch.nn.functional.gelu(x)
            x = self.layer_norm(x)

            mask = (labels != -100)
            expanded_mask = mask.unsqueeze(-1).expand(-1, -1, hidden_size)
            selected_slices = x[expanded_mask].view(-1, hidden_size)

            # project back to size of vocabulary with bias
            x = self.decoder(selected_slices)
            return x
        
        self.model.lm_head.forward = lm_head_forward

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        outputs = self.model.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        labels = labels.to(sequence_output.device)
        prediction_scores = self.model.lm_head(sequence_output, labels)
        mask = (labels != -100)
        flat_labels = labels[mask]
        # Calculate loss only on masked tokens
        masked_lm_loss = self.loss_fn(prediction_scores, flat_labels)

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
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
        concat_ds = load_dataset("awidjaja/512-food-dataset", split="train").take(2048)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dataset = tokenize(concat_ds, self.tokenizer)
        self.dataset.set_format("torch", columns=["input_ids", "attention_mask"])
        self.dataset = self.dataset.train_test_split(test_size=VAL_SIZE)
        self.train_dataset = self.dataset['train']
        self.val_dataset = self.dataset['test']
        self.train_dataset = concatenate_datasets([self.train_dataset] * DUPLICATE)
        self.val_dataset = concatenate_datasets([self.val_dataset] * DUPLICATE)
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=MLM_PROB)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=True
            )
            self.val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=False
            )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=self.data_collator,
            sampler=self.train_sampler,
            num_workers=NUM_WORKERS
        )
        return pl.MpDeviceLoader(train_loader, xm.xla_device())

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=self.data_collator,
            sampler=self.val_sampler,
            num_workers=NUM_WORKERS
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
        accelerator=ACCELERATOR,
        max_epochs=EPOCH,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=4,
        precision='16-true'
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
