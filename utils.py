import torch
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
import json
import os
import pdb
import numpy as np
from sklearn.metrics import accuracy_score

import sys
sys.path.append('mediqa-m3g-startingkit-v2')
#modified from https://github.com/mgalley/sacreBLEU
import sacrebleu_deltableu

try:
    import evaluate
    import pandas as pd
except ImportError as e:
    print( e )
    pass  # module doesn't exist, deal with it.


### INFERENCE SEARCH ALGORITHMS 

def treebank_tokenize(s):
    return TreebankWordTokenizer().tokenize(s)

def greedy_search_inference(
    model,
    tokenizer,
    beam_size: int = 1,
    generated=None,
    entry_length=8,
    temperature=1.0,
    stop_token: str = "<|endoftext|>",
):
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    
    generated_tokens = []
    with torch.no_grad():
        for _ in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits

            logits = logits[:, -1:, :] / (temperature if temperature > 0 else 1.0) # picking only the last sequence
            
            logits = logits.softmax(-1).log()
            # final_logit
            
            _ , next_tokens = logits.topk(beam_size, -1)
            # pdb.set_trace()
            
            next_tokens = next_tokens.permute(0, 2, 1)
            # pdb.set_trace()
            generated_tokens.append(next_tokens.squeeze(1))
            
            if model.gpttype == "microsoft/biogpt":
                next_token_embed = model.gpt.biogpt.embed_tokens(
                    next_tokens.squeeze(1)
                )# .view(generated.shape[0], 1, -1)
            elif model.gpttype == "gpt2-xl":
                next_token_embed = model.gpt.transformer.wte(
                    next_tokens.squeeze(1)
                )# .view(generated.shape[0], 1, -1)
            else:
                next_token_embed = model.gpt.get_input_embeddings()(tokens[:,-1])
                next_token_embed=next_token_embed.squeeze(1) # .view(generated.shape[0], 1, -1)
            # pdb.set_trace()
            generated = torch.cat((generated, next_token_embed), dim=1)
            
            # Check if any of the generated tokens match the stop token
            if (next_tokens == stop_token_index).all():
                break
            # pdb.set_trace()
    
    
    # Stack all generated tokens to get the final output
    generated_tokens = torch.cat(generated_tokens, dim=1)  # (batch_size, generated_length)

    # Decode the tokens into text for each batch element
    output_texts = [model.tokenizer.decode(generated_tokens[i].tolist(), skip_special_tokens=True) for i in range(generated.shape[0])]
    # pdb.set_trace()
    return output_texts

def beam_search_inference(
    model,
    tokenizer,
    beam_size: int = 5,
    generated=None,
    entry_length=16,
    temperature=1.0,
    stop_token: str = "<|endoftext|>",
):
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones((generated.shape[0], beam_size), device=device)
    # is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    is_stopped = torch.zeros((generated.shape[0], beam_size), device=device, dtype=torch.bool)
    with torch.no_grad():
        for _ in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits

            logits = logits[:, -1:, :] / (temperature if temperature > 0 else 1.0) # picking only the last sequence
            # logits = logits[:, :, :] / (temperature if temperature > 0 else 1.0)

            logits = logits.softmax(-1).log()
            # final_logit
            pdb.set_trace()
            # if scores is None:
            scores, next_tokens = logits.topk(beam_size, -1)
            # pdb.set_trace()
            # generated = generated.expand(beam_size, *generated.shape[1:])     # create "beam_size" copies of a generated input embeddings
           
            # next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
            next_tokens, scores = next_tokens.permute(0, 2, 1), scores # .permute(0, 2, 1)
            if tokens is None:
                tokens = next_tokens
            else:
                pdb.set_trace()
                tokens = tokens.expand(beam_size, *tokens.shape[1:])
                tokens = torch.cat((tokens, next_tokens), dim=1)
            generated = generated.unsqueeze(1).expand(-1, beam_size, -1, -1)  # Shape: (2, 5, 232, 1600) # account for batch size (batch, beam, seq, vocab)
            generated = generated.reshape(generated.shape[0] * beam_size, *generated.shape[2:]) # reshape to (batch * beam_size, seq, vocab)
            # else:
            #     pdb.set_trace()
            #     # logits[is_stopped] = -float(np.inf)
            #     # logits[is_stopped, 0] = 0
            #     scores_sum = scores.squeeze(1) # + logits
            #     seq_lengths[~is_stopped] += 1
            #     scores_sum_average = scores_sum / seq_lengths # [:, None]
            #     scores_sum_average, next_tokens = scores_sum_average.topk(beam_size, -1) # .view(-1).topk(beam_size, -1)
            #     next_tokens_source = next_tokens // scores_sum.shape[1]
            #     seq_lengths = seq_lengths[next_tokens_source]
            #     next_tokens = next_tokens % scores_sum.shape[1]
            #     next_tokens = next_tokens.unsqueeze(1)
            #     pdb.set_trace()
            #     tokens = tokens[next_tokens_source]
            #     tokens = torch.cat((tokens, next_tokens), dim=1)
            #     generated = generated[next_tokens_source]
            #     scores = scores_sum_average * seq_lengths
            #     is_stopped = is_stopped[next_tokens_source]
            if model.gpttype == "microsoft/biogpt":
                next_token_embed = model.gpt.biogpt.embed_tokens(
                    next_tokens.squeeze()
                )# .view(generated.shape[0], 1, -1)
            elif model.gpttype == "gpt2-xl":
                next_token_embed = model.gpt.transformer.wte(
                    next_tokens.squeeze()
                )# .view(generated.shape[0], 1, -1)
            else:
                next_token_embed = model.gpt.get_input_embeddings()(tokens[:,-1])
                next_token_embed=next_token_embed.squeeze().view(generated.shape[0], 1, -1)
            # pdb.set_trace()
            # generated = torch.cat((generated, next_token_embed), dim=1)
            generated = torch.cat((generated, next_token_embed.unsqueeze(2)), dim=2) # add seq dimension and concatenate on seq dimension
            if generated.dim() == 4:
                generated = generated.view(-1, generated.size(2), generated.size(3))
            # pdb.set_trace()
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze(2)
            if is_stopped.all():
                break
    pdb.set_trace()
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts

def print_nearest_text_token(vis_token, model):
    """print the nearest token in the vocabulary to the given token through model.gpt.embeddings.weight"""
    embeddings = model.gpt.transformer.wte.weight
    distances = torch.norm(embeddings - vis_token, dim=1)
    nearest_token_idx = torch.argmin(distances)
    print(model.tokenizer.decode([nearest_token_idx.item()])) 

def get_existing_encounter_idx(output, encounter_id):
    for i in range(len(output)):
        if output[i]['encounter_id'] == encounter_id:
            return i


### EVALUATION METRICS
print('Checking BLEU')
bleu = evaluate.load("sacrebleu")
# https://huggingface.co/spaces/evaluate-metric/sacrebleu
# >>> predictions = ["hello there", "general kenobi"]
# >>> references = [["hello", "there"], ["general kenobi", "general yoda"]]
# >>> sacrebleu = evaluate.load("sacrebleu"
# >>> results = sacrebleu.compute(predictions=predictions,
# ...                         references=references)
# will also run: https://github.com/mgalley/sacreBLEU
# #############


def read_json( file_path ) :
    with open( file_path, 'r' ) as f :
        return json.load( f )


def get_weight_schemes( df_userinfo, author_id, completeness, contains_freq_ans ) :

    userrow = df_userinfo[ df_userinfo['author_id']==author_id ].iloc[0]
    user_validlevel = userrow[ 'validation_level' ]
    user_rank = userrow[ 'rank_level' ]
    if user_rank == 'unk' :
        user_rank = 'level_0'

    answer_profile = [
           [ 1 if str(user_validlevel)[:2]=='md' else 0 ][0],
           [ 1 if int( user_rank.replace('level_','') ) >=4 else 0 ][0],
           [ 1 if contains_freq_ans==1.0 else 0 ][0],
           [ 1 if completeness==1.0 else 0 ][0],
        ]
    
    w1 = -100
        
    if answer_profile[-1] == 1 :
        first_3 = sum( answer_profile[:-1] )
        if first_3 == 3 :
            w1 = 1.0
        elif first_3 == 2 :
            w1 = 0.9
        elif first_3 == 1 :
            w1 = 0.8
        else :
            w1 = 0.7
    else :
        first_3 = sum( answer_profile[:-1] )
        if first_3 == 3 :
            w1 = 0.9
        elif first_3 == 2 :
            w1 = 0.8
        elif first_3 == 1 :
            w1 = 0.7
        else :
            w1 = 0.6

    return w1


def get_ref_scores( references, df_userinfo ) :

    ref_scores = []

    for reference in references :
        author_id = reference[ 'author_id' ]
        completeness = reference[ 'completeness' ]
        contains_freq_ans = reference[ 'contains_freq_ans' ]
        ref_scores.append( get_weight_schemes( df_userinfo, author_id, completeness, contains_freq_ans ) )

    return ref_scores


def get_deltableu_score(prediction_dir, reference_dir = 'mediqa-m3g-startingkit-v2/mediqa-m3-clinicalnlp2024/valid_ht.json', \
                        score_dir = 'mediqa-m3g-startingkit-v2/app/output/', user_info_dir = 'mediqa-m3g-startingkit-v2/app/input/ref/df_userinfo.csv',
                        out_filename = 'scores'):
    
    truth = read_json( reference_dir )
    reference_langs = [ x.replace( 'content_', '' ) for x in truth[0]['responses'][0].keys() if 'content_' in x ]
    reference_ids = [ x['encounter_id'] for x in truth ]
    prediction = read_json( prediction_dir )
    prediction_langs = [ x.replace( 'content_', '' ) for x in prediction[0]['responses'][0].keys() if 'content_' in x ]
    prediction_ids = [ x['encounter_id'] for x in prediction ]
    
    bad_match = 0
    for ind, reference_id in enumerate( reference_ids ) :
        prediction_id = prediction_ids[ ind ]
        if reference_id != prediction_id :
            bad_match += 1
            print( 'INDEX {} has different ids for reference and prediction, {} and {} respectively.'.format( ind, reference_id, prediction_id ) )

    if bad_match > 0 :
        print('Please check that your encounter id for your prediction and input are in the same order!!')
        sys.exit(0)

    prediction_langs = list( set( prediction_langs ) & set( reference_langs ) )

    references = {}
    predictions = {}
    reference_weights = []

    for prediction_lang in prediction_langs :
        references[ prediction_lang ] = []
        predictions[ prediction_lang ] = []

    df_userinfo = pd.read_csv( user_info_dir )

    for ind, reference_instance in enumerate( truth ) :
        
        reference_weights.append( get_ref_scores( reference_instance[ 'responses' ], df_userinfo ) )
        for prediction_lang in prediction_langs :
            
            refs = [ x[ 'content_{}'.format( prediction_lang ) ] for x in reference_instance[ 'responses' ] ]
            hyp = prediction[ ind ][ 'responses' ][0][ 'content_{}'.format( prediction_lang ) ]
            
            references[ prediction_lang ].append( refs )
            predictions[ prediction_lang ].append( hyp )

    scores = {}

    for pred_lang in prediction_langs :

        if pred_lang == 'zh' :
            delatbleu = sacrebleu_deltableu.corpus_bleu_t( predictions[ pred_lang ],
                                            references[ pred_lang ],
                                            ref_weights= reference_weights,
                                            tokenize='zh' )
        else :
            delatbleu = sacrebleu_deltableu.corpus_bleu_t( predictions[ pred_lang ],
                                            references[ pred_lang ],
                                            ref_weights= reference_weights )
        print( delatbleu )
        scores[ 'deltableu_{}'.format( pred_lang) ] = delatbleu.score


    print(scores)

    mean_bleu = np.mean(list(scores.values()))

    with open(os.path.join(score_dir, f'{out_filename}.json'), 'a') as score_file:
        score_file.write(json.dumps(scores) + '\n')

    return mean_bleu