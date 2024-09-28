from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from tqdm import tqdm
import copy
import os
import numpy as np
import time
import random
import torch.nn as nn
import torch.nn.functional as nnf
import os
import numpy as np
import random
import pandas as pd
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from torch.nn import functional as nnf
from accelerate import Accelerator
from utils import (
    # generate_beam,
    greedy_search_inference,
    # print_nearest_text_token,
    get_existing_encounter_idx,
    get_deltableu_score,
    # beam_search_inference
)

from transformers import AutoTokenizer, AutoModelForCausalLM
import json

def society_inference(dataloader, ensemble, args, reference_dir='mediqa-m3g-startingkit-v2/app/input/ref/reference_test.json', leader_name = "mistralai/Mistral-7B-v0.1"):
    split = args.split
    ## using accelerator 
    accelerator = Accelerator()
    device = accelerator.device

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    # leader_name = "mistralai/Mistral-7B-v0.1"  # LLM to summarize answers
    
    leader_tokenizer = AutoTokenizer.from_pretrained(leader_name)
    leader_model = AutoModelForCausalLM.from_pretrained(leader_name)
    leader_model.half()
    # model = model_obj.to(device)

    ## 💡 introduce all components to accelerate library
    dataloader, leader_tokenizer, leader_model = accelerator.prepare(
        dataloader, leader_tokenizer, leader_model
    )
    ensemble = [model.half().to(device).eval() for model in ensemble]

    leader_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(leader_model, inplace=True)
    
    accelerator.wait_for_everyone()

    # model.eval()
    existing_encounter_ids = []
    output = []
    
    

    # Wrap dataloader with tqdm for progress tracking
    for i, (prefix, tokens, mask, q_len) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing Batches"):
        # if i >= 2:  # Stop after processing the first two batches for debugging
        #     break
        torch.cuda.empty_cache()

        prefix = prefix.type(torch.float16)
        prefix = prefix.squeeze(0)
        tokens = tokens.type(torch.long)
        tokens = tokens.squeeze(0)
        mask = mask.type(torch.float16)
        mask = mask.squeeze(0)
        q_len = q_len.type(torch.int16)

        question = ensemble[0].tokenizer.decode(tokens[0, :q_len - 20])
        answers_per_model = []
        for model in ensemble:

            with torch.no_grad():
                embed = model.generate(prefix, tokens, mask, q_len)
                batch_answers = greedy_search_inference(model, model.tokenizer, generated=embed, entry_length=16, temperature=1)
                answers_per_model += batch_answers
        ### Leader summarization
        prompt = f"Eres el líder experto de un equipo de dermatólogos. Tienes el siguiente caso: {question} Tus colegas ofrecen las siguientes respuestas: "
        for answer in answers_per_model:
            prompt += f"- {answer}\n"
        
        prompt += "\n Ofrece un resumen conciso de las respuestas: "
        
        # Tokenize the prompt
        inputs = leader_tokenizer(prompt, return_tensors="pt", max_length=16, truncation=True).to(prefix.device)
        
        # Generate summary using the model
        
        input_ids = inputs['input_ids'].to(torch.int16)
        att_mask_32 = inputs['attention_mask'].to(torch.int16)
        summary_ids = leader_model.generate(input_ids, attention_mask = att_mask_32, max_new_tokens=2, num_beams=1) # do greedy search for computational efficiency on cpu
        summary = leader_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        # pdb.set_trace()
        encounter_id = dataloader.dataset.data[i][0][0]
        if not (encounter_id in existing_encounter_ids):
            output.append({
                "encounter_id": encounter_id,
                "responses": [{}]
            })
        existing_encounter_ids.append(encounter_id)

        encounter_idx = get_existing_encounter_idx(output=output, encounter_id=encounter_id)

        output[encounter_idx]["responses"][dataloader.dataset.data[i][0][1]] = summary
        
        # # Remove the last comma and close the JSON array
        # f.seek(f.tell() - 2)  # Move cursor back to overwrite last comma
        # f.write("\n]")  # End of JSON array
        output_json = json.dumps(output, indent=4)
    with open(f"output_{split}.json", "w") as f:
        f.write(output_json)
    
    total_sacrebleu = get_deltableu_score(f'output_{split}.json', reference_dir=reference_dir)


    return total_sacrebleu
