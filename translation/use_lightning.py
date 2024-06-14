from lightning import LightningModule, LightningDataModule, Trainer
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from torch import nn, optim
from lightning.pytorch.loggers import WandbLogger
import torch
import os
import wandb

from huggingface_hub import login
login(os.getenv("ACCESS_TOKEN"))

num_chips = int(os.getenv("NUM_CHIPS","16"))

WEIGHT_DECAY = 0.001
LEARNING_RATE = 1e-5
VAL_SIZE = 0.05
EPOCH = 3
BATCH_SIZE = 8
block_size = 256

HUB_MODEL_NAME="thonyyy/qwen2-7b-translate-p2skip"
DATASET_NAME="thonyyy/tatoeba-nusax-mt-p2"
ACCELERATOR="tpu"
BASE_MODEL_NAME="Qwen/Qwen2-7B-Instruct"
NUM_WORKERS = 32

effective_batch_size = 2048
gradient_acummulation_steps = effective_batch_size // (num_chips*BATCH_SIZE)

def formatting_prompts_func(dataset, tokenizer):
    def _map_func(examples):
        prompt_list = []
        for source_lang, target_lang, source_text, target_text in zip(examples['source_lang'],examples['target_lang'],examples['source_text'],examples['target_text']):
            source_text = tokenizer.decode(tokenizer(source_text, add_special_tokens = False).input_ids[:(block_size-80)//2])
            target_text = tokenizer.decode(tokenizer(target_text, add_special_tokens = False).input_ids[:(block_size-80)//2])
            messages = [
                {"role": "system", "content": "You are a professional translator capable of translating English, Indonesian, and Indonesian Regional Languages"},
                {"role": "user", "content": f"""Translate this text from {source_lang} into {target_lang}:\n{source_text}"""},
                {"role": "assistant", "content": target_text},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            prompt_list.append(text)
        outputs = tokenizer(
            prompt_list,
            max_length = block_size,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt",
        )
        return outputs
    remove_cols = dataset.column_names
    return dataset.map(_map_func,
                       batched = True,
                       num_proc = NUM_WORKERS,
                       remove_columns = remove_cols
                       )

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
        dataset = load_dataset(DATASET_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_size='left')
        self.dataset = dataset['train'].train_test_split(test_size=VAL_SIZE, seed = 42)
        self.train_dataset = self.dataset['train'].shuffle(seed = 42)
        self.val_dataset = self.dataset['test'].shuffle(seed = 42)
        self.train_dataset = formatting_prompts_func(self.train_dataset, self.tokenizer)
        self.val_dataset = formatting_prompts_func(self.val_dataset, self.tokenizer)
        from trl import DataCollatorForCompletionOnlyLM
        response_template = "<|im_start|>assistant\n"
        self.data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)
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
    TOTAL_STEPS = len(data.train_dataset)
    WARMUP_STEPS = int(0.1*TOTAL_STEPS)
    model = LargeLanguageModel(model_name = BASE_MODEL_NAME, total_steps = TOTAL_STEPS, warmup_steps = WARMUP_STEPS, tokenizer = data.tokenizer)
    wandblogger = WandbLogger(
        log_model = True,
        mode = "online",
        project = "qwen-for-translation-1",
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
        logger = wandblogger,
        precision = 'bf16-true',
        accumulate_grad_batches = gradient_acummulation_steps
    )
    trainer.fit(model, data)
    model.model.push_to_hub(HUB_MODEL_NAME, private = True)
    data.tokenizer.push_to_hub(HUB_MODEL_NAME, private = True)
        
    wandb.finish()
    
if __name__ == "__main__":
    main()
