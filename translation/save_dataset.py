import os
import pandas as pd

def read_lines(filepath):
    with open(filepath, 'r') as file:
        res = file.read().split('\n')
    if res[-1] == '':
        res = res[:-1]
    return res

import pandas as pd
frame= []
absolute_path = r"data/release/v2023-09-26/"
for subset in ['ban-msa','eng-jav','eng-msa','eng-sun','jav-msa','msa-msa']:
    for split in ['dev','test','train']:
        try:
            df = pd.read_csv(rf"{absolute_path}/{subset}/{split}.id", sep="\t" , names = ['data_source','source_lang','target_lang'] if split =='train' else ['source_lang','target_lang'])
            src_lines = read_lines(rf"{absolute_path}/{subset}/{split}.src")
            trg_lines = read_lines(rf"{absolute_path}/{subset}/{split}.trg")
            df = df[['source_lang','target_lang']]
            df['source_text'] = src_lines
            df['target_text'] = trg_lines
            df['data_split'] = split
            df['data_source'] = "tatoeba-"+subset
            
            df = df[(df['source_text'] != '') & (df['target_text'] != '')]
            df = df[df['source_lang'] != df['target_lang']]
            frame.append(df)
            print(f"Dataset {subset} split {split} successfully added with {len(df)} rows")
        except:
            print(f"Dataset {subset} split {split} doesn't exist")

df = pd.concat(frame)
del frame

mapp = {
    "ban":"balinese",
    "ind":"indonesian",
    "bjn":"banjarese",
    "min":"minangkabau",
    "jav":"javanese",
    # "zlm":"malaysian malay",
    "sun":"sundanese",
    "max_Latn":"north maluku malay",
    "zlm_Latn": "malaysian malay",
    "zsm_Latn": "malaysian malay",
    # 'tmw_Latn': "temuan malay",
    'msa_Latn' : "indonesian malay",
    #'msa' : "indonesian malay",
    "eng" : "english",
    # 'jak_Latn: "jakun malay'
 }

dfff = df[(df['target_lang'].isin(mapp.keys())) & (df['source_lang'].isin(mapp.keys()))]
del df
dfff['target_lang'] = dfff['target_lang'].map(mapp)
dfff['source_lang'] = dfff['source_lang'].map(mapp)

## NUSA X

base_folder = "nusax/datasets/mt"
import os
data_dict = {}

for file_name in os.listdir(base_folder):
  if file_name.endswith('.csv'):
    split_type = file_name.split(".")[0]
    path_file = os.path.join(base_folder,file_name)
    dff = pd.read_csv(path_file,index_col=False)
    dff.drop(columns = ['Unnamed: 0'], inplace=True)
    cols = list(dff.columns)
    for idx,col in enumerate(cols):
      for idx_s,col_s in enumerate(cols):
        if idx_s != idx:
          df_filt = dff[[col,col_s]].copy()
          df_filt.rename(columns={f'{col}':"source_text",f'{col_s}':"target_text"}, inplace=True)
          df_filt["source_lang"] = col
          df_filt["target_lang"] = col_s
          df_filt["data_split"] = split_type
          df_filt["data_source"] = "nusax-mt"
          key_data = f"{col}_{col_s}"
          if(key_data in data_dict.keys()):
            temp_df = data_dict[key_data]
            data_dict[key_data] = pd.concat([temp_df,df_filt],axis=0)
          else:
            data_dict[key_data] = df_filt

df_concat = pd.DataFrame()
for key in data_dict.keys():
  dff = data_dict[key]
  if(df_concat.empty):
    df_concat = dff
  else:
    df_concat = pd.concat([df_concat,dff],axis=0)

base_folder = "nusax/datasets/lexicon"
import os
P = []
lang_list = os.listdir(base_folder)
for lang in lang_list:
    p = pd.read_csv(f"{base_folder}/{lang}")
    lang = lang.replace(".csv","")
    p = p.rename(columns = {'indonesian':'source_text',lang:"target_text"})
    p['source_lang'] = "indonesian"
    p['target_lang'] = lang
    p['data_source'] = 'nusax-lexicon'
    p['data_split'] = split
    P.append(p)
df_p = pd.concat(P)

from datasets import Dataset
from huggingface_hub import login
login(os.getenv("ACCESS_TOKEN"))
df = pd.concat([dfff, df_concat,df_p])
del dfff, df_concat
ds = Dataset.from_pandas(df.reset_index(drop = True))
del df
ds.push_to_hub("thonyyy/tatoeba-nusax-mt")

base_folder = "nusax/datasets/sentiment"
import os
P = []
lang_list = os.listdir(base_folder)
for lang in lang_list:
    for split in ['train',"test","valid"]:
        p = pd.read_csv(f"{base_folder}/{lang}/{split}.csv")
        p['language'] = lang
        p['split'] = split
        P.append(p)
        
df_p = pd.concat(P)
df_p.to_csv("nusax_sentiment.csv")
ds = Dataset.from_pandas(df_p.reset_index(drop = True))
ds.push_to_hub("thonyyy/nusax_sentiment")
