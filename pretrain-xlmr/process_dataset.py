import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
from huggingface_hub import login
import os
login(os.getenv("ACCESS_TOKEN"))

def tokenize_and_split_document(document, max_length=512):
    n = len(tokenizer.encode(document))
    if n <= max_length:
        return [document]
    else:
        sentences = sent_tokenize(document)
        m = len(sentences)
        tokenized_sentences = [tokenizer.encode(sentence, add_special_tokens=False) for sentence in sentences]
        
        tokenized_document = []
        current_chunk = []
        idx = 0

        for i,tokenized_sentence in enumerate(tokenized_sentences):
            if len(current_chunk) + len(tokenized_sentence) <= max_length:
                current_chunk.extend(tokenized_sentence)
            else:
                tokenized_document.append(current_chunk)
                current_chunk = tokenized_sentence
                idx = i

        if current_chunk:
            try:
                while len(current_chunk) < max_length//2:
                    current_chunk = tokenized_sentences[idx-1] + current_chunk
                    idx -= 1
            except:
                pass
            else:
                tokenized_document.append(current_chunk)
        
        # Decode chunks back to text
        split_docs = [tokenizer.decode(chunk, clean_up_tokenization_spaces=True) for chunk in tokenized_document]
        return split_docs

def split_dataset(examples):
    sentences = []
    for s in examples['content']:
        sentences.extend(tokenize_and_split_document(s))
    return {"text":sentences}

list_dataset = "awidjaja/just_another_food_wikipedia_dataset,awidjaja/just_another_recipe_dataset,awidjaja/just_a_food_wikipedia_dataset,awidjaja/just_a_recipe_dataset".split(",")
dataset = concatenate_datasets([load_dataset(ds_name, split="train") for ds_name in list_dataset])
dataset = dataset.map(split_dataset, batched=True, num_proc=512, remove_columns=["content","split","title","source"])
dataset.push_to_hub("awidjaja/512-food-dataset")

