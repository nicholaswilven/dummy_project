
source_text, source_lang, target_lang, target_text = "indo","indo","indo","indo"

x = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    Translate the input from {source_lang.replace('_',' ').title()} to {target_lang.replace('_',' ').title()}.

    ### Input:
    {source_text}

    ### Response:
    {target_text} </s>"""

if __name__=="__main__":          
    print(x)