from lightning import LightningModule, LightningDataModule, Trainer
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, IterableDataset
from torch import nn, optim
from lightning.pytorch.loggers import WandbLogger
import torch
import os
import wandb

from huggingface_hub import login
login(os.getenv("ACCESS_TOKEN"))

WEIGHT_DECAY=0.001
LEARNING_RATE=1e-5
VAL_SIZE=0.05
EPOCH=2
BATCH_SIZE = 8
block_size = 256
HUB_MODEL_NAME="thonyyy/qwen2-7b-translate-p1"
DATASET_NAME="thonyyy/tatoeba-nusax-mt-p1-stream"
ACCELERATOR="tpu"
BASE_MODEL_NAME="Qwen/Qwen2-7B-Instruct"
NUM_WORKERS = 32
MAX_TRAIN_BATCHES_PER_EPOCH = 192000000//(BATCH_SIZE*32) # 32 parallel process

class StreamingIterableDataset(IterableDataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.EOS_TOKEN = self.tokenizer.eos_token
        self.alpaca_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{}\n\n### Response:\n{}"
    
    def __iter__(self):
        for example in self.dataset:
            source_lang = example['source_lang']
            target_lang = example['target_lang']
            source_text = self.tokenizer.decode(self.tokenizer(example['source_text'], add_special_tokens=False).input_ids[:(block_size-80)//2])
            target_text = self.tokenizer.decode(self.tokenizer(example['target_text'], add_special_tokens=False).input_ids[:(block_size-80)//2])
            instruction = f"Translate the input text from {source_lang.replace('_',' ').title()} to {target_lang.replace('_',' ').title()}."
            prompt = self.alpaca_prompt.format(instruction+'\n'+source_text.replace('\n',''), target_text.replace('\n','')) + self.EOS_TOKEN
            
            outputs = self.tokenizer(
                prompt,
                max_length = block_size,
                padding = "max_length",
                truncation = True,
                return_tensors = "pt",
            )
            yield {key: val.squeeze() for key, val in outputs.items()}

class LargeLanguageModel(LightningModule):
    def __init__(self, model_name: str = "", total_steps = 0, warmup_steps = 0, tokenizer = None):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
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
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj",],
            inference_mode=False,
            r=32,
            lora_alpha=16,
            lora_dropout=0.1
        )
        self.model = get_peft_model(self.model, lora_config)
        print("LoRA enabled")
        self.model.print_trainable_parameters()
        
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.train_dataset = load_dataset(DATASET_NAME, split='train', streaming=True)
        self.val_dataset = load_dataset(DATASET_NAME, split='validation', streaming=True)
        from trl import DataCollatorForCompletionOnlyLM
        response_template = "### Response:\n"
        self.data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)
    
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        seed, buffer_size = 42, 10_000
        self.train_dataset = self.train_dataset.shuffle(seed, buffer_size=buffer_size)
        train_dataset = StreamingIterableDataset(self.train_dataset, self.tokenizer)
        return DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=self.data_collator)
    
    def val_dataloader(self):
        val_dataset = StreamingIterableDataset(self.val_dataset, self.tokenizer)
        return DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=self.data_collator)

def main():
    data = LargeDataModule(model_name=BASE_MODEL_NAME)
    TOTAL_STEPS = MAX_TRAIN_BATCHES_PER_EPOCH * EPOCH
    WARMUP_STEPS = int(0.1 * TOTAL_STEPS)
    model = LargeLanguageModel(model_name=BASE_MODEL_NAME, total_steps=TOTAL_STEPS, warmup_steps=WARMUP_STEPS, tokenizer=data.tokenizer)
    wandblogger = WandbLogger(
        log_model = True,
        mode = "online",
        project = "qwen2-for-translation-p1",
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
        max_steps = TOTAL_STEPS,
        logger = wandblogger,
        precision = 'bf16-true'
    )
    trainer.fit(model, data)
    model.model.push_to_hub(HUB_MODEL_NAME, private = True)
    data.tokenizer.push_to_hub(HUB_MODEL_NAME, private = True)
        
    wandb.finish()

if __name__ == "__main__":
    main()
