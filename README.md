# random-nlp-things
Pytorch Lightning implementation for various NLP tasks
# Code
This repository contains:
1. Finetuning Transformer decoder (Llama/Komodo/Qwen) for Indonesian Regional Language Translations, modes: Vanilla finetuning, Low Rank Adaptation (LoRA) via peft, Fully Sharded Data Parallel (FSDP) with Pytorch XLA.
2. Additional pretraining script for Transformer encoder (XLM Roberta) with Masked Language Modelling Objective.
3. Finetuning Transformer encoder (XLM Roberta) for zero shot classification.
4. Finetuning Transformer encoder (XLM Roberta) for sentiment analysis.
