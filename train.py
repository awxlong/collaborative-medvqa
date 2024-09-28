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


def train(train_loader, valid_loader, model_obj, args):
    ## 💡 using accelerator 
    accelerator = Accelerator()
    device = accelerator.device

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    model = model_obj.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)


    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.epochs * len(train_loader),
    )

    ## 💡 introduce all components to accelerate library
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    valid_loader = accelerator.prepare(valid_loader)

    model.half()
    best_valid_loss = float("inf")
    counter = 0
    n_epochs = args.epochs
    accumulation_steps = args.grad_acc_steps
    accelerator.wait_for_everyone()
    for epoch in range(args.epochs):

        with tqdm(total=args.batch_size * len(train_loader)) as epoch_pbar:
            epoch_pbar.set_description(f"Epoch {epoch}")
            start_time = time.time()
            model.train()
            total_loss = 0.0
            total_acc = 0.0
            total_rocauc = 0.0

            for i, (prefix, tokens, mask, q_len) in enumerate(train_loader):
                with accelerator.accumulate(model):
                    prefix = prefix.type(torch.float16)
                    prefix = prefix.squeeze(0)
                    tokens = tokens.type(torch.long)
                    tokens = tokens.squeeze(0)
                    mask = mask.type(torch.float16)
                    mask = mask.squeeze(0)
                    q_len = q_len.type(torch.int16)
                    # 
                    outputs = model(prefix, tokens, mask, q_len, batch_size=prefix.shape[0])
                    # pdb.set_trace()
                    logits = outputs.logits
                    loss = 0.

                    shift = 10 if args.setting=="p_tuning" or args.setting=="prompttuning" else 0 

                    for b in range(logits.size(0)):
                        
                        condensed_tokens = tokens[b, q_len + model.prefix_length+1:]
                        # condensed_tokens = condensed_tokens.type(torch.float16)
                        condensed_logits = logits[b, shift + q_len + model.prefix_length:-1]
                        condensed_logits = condensed_logits.to(torch.float32)
                        # pdb.set_trace()
                        loss+= nnf.cross_entropy(condensed_logits.reshape(-1,logits.shape[-1]), condensed_tokens.flatten(), ignore_index=0)
                    loss=loss/logits.size(0)    

                    accelerator.backward(loss)
                    if (i + 1) % accumulation_steps == 0:
                        optimizer.step()  # Update weights
                        scheduler.step()
                        optimizer.zero_grad()  # Reset gradients
                    
                    total_loss += loss.item()
                    avg_loss = total_loss / (i+1)
                    
                    desc = f"Epoch {epoch} - loss {avg_loss:.20f}"
                    epoch_pbar.set_description(desc)
                    epoch_pbar.update(prefix.shape[0])

        # torch.cuda.empty_cache()
        model.eval()

        total_loss = 0.0
        
        # total_rocauc = 0.0
        with tqdm(total=args.batch_size * len(valid_loader)) as epoch_pbar:
            epoch_pbar.set_description(f"VAL Epoch {epoch}")
            for i, (prefix, labels, tokens, mask,q_len) in enumerate(valid_loader):
                torch.cuda.empty_cache()
                
                prefix = prefix.type(torch.float16)
                prefix = prefix.squeeze(0)
                tokens = tokens.type(torch.long)
                tokens = tokens.squeeze(0)
                mask = mask.type(torch.float16)
                mask = mask.squeeze(0)
                q_len = q_len.type(torch.int16)
                with torch.no_grad():
                    outputs = model(prefix, labels, tokens, mask, q_len, batch_size=prefix.shape[0])
                    logits = outputs.logits
                    loss = 0.
                    shift = 10 if args.setting=="p_tuning" or args.setting=="prompttuning" else 0 
                    for b in range(logits.size(0)):
                        condensed_tokens = tokens[b, q_len + model.prefix_length+1:]
                        condensed_logits = logits[b, shift + q_len + model.prefix_length:-1]
                        condensed_logits = condensed_logits.to(torch.float32)
                        # pdb.set_trace()
                        loss+= nnf.cross_entropy(condensed_logits.reshape(-1,logits.shape[-1]), condensed_tokens.flatten(), ignore_index=0)
                    loss=loss/logits.size(0)    
                    total_loss += loss.item()
                avg_val_loss = total_loss / (i + 1)
                desc = f"VAL Epoch {epoch} - loss {avg_val_loss:.20f}"
                epoch_pbar.set_description(desc)
                epoch_pbar.update(prefix.shape[0])

        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss

            torch.save(model.state_dict(), os.path.join(args.out_dir, f"open_ended_latest.pt"))

        scheduler.step()
        elapsed_time = time.time() - start_time
        print(
            "VAL epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s".format(
                epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time
            )
        )
        if avg_val_loss > avg_loss:
            counter += 1
        if counter == 5:
            break
    return model