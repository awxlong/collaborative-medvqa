# import requests
import random
import os
import json
from PIL import Image
import cv2
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering,BlipImageProcessor, AutoProcessor
from transformers import BlipConfig
# from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import gc
import torch.nn as nn
from typing import List
from numpy import ndarray

import sys
from transformers import AutoTokenizer
from torch.utils.data import Dataset
# import pickle
from CFG import CFG

import pdb



def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # Ensure loading as RGB image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB if needed
    img = img.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    return img
   
    
class BuildVQADatasetSingleLanguage(torch.utils.data.Dataset):
    '''
    Loading VQA dataset for a single language
    Args:
        json_file - str - directory where train.json file can be found
        train_dir - str - directory where images for training, val and testing are found
        transforms - Albumentations - image transformations defined in CFG file
        train - bool - flag indicating whether to load images, Q&A or only images and questions
        long_answer_ratio - float - number indicating preference for longer answers during bootstrapping
        language - str - indicates which language to load
    Returns:
        encounter_id - str - unique ID for writing into prediction.json file
        images - torch.Tensor - stack of dermatology images
        question - str - unique patient's question
        answers - List[str] - list of answers given by different doctors
        
        encounter_id, images, question, answers if train = True
        encounter_id, images, question if train = False

    '''
    def __init__(self,json_file='/kaggle/input/mediqa-m3g/mediqa-m3-clinicalnlp2024/train.json', train_dir='/kaggle/input/mediqa-m3g/images_final/images_train', 
                 transforms=None, train=True, long_answer_ratio = 0.7, language = 'content_en'):
        with open(json_file) as f:
            self.qa_pairs = json.load(f)
        self.transforms = transforms
        self.train_dir = train_dir
        self.load_encounter_id = lambda qa_pairs: qa_pairs['encounter_id']
        self.load_image_paths = lambda encounter_id: [os.path.join(self.train_dir, img_id) for img_id in encounter_id]
        self.load_questions = lambda qa_pairs: qa_pairs[f'query_{language}']
        self.train = train
        self.load_answers = lambda qa_pairs: [answer[f'{language}'] for answer in qa_pairs['responses']] if self.train else None
        self.long_answer_ratio = long_answer_ratio # for bootstrapping longer answers
        self.language = language
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, index):
        qa_pair = self.qa_pairs[index]

        img_paths = self.load_image_paths(qa_pair['image_ids'])
        images = [load_img(img_path)  for img_path in img_paths]
        question: str = self.load_questions(qa_pair)
        answers: List[str] = self.load_answers(qa_pair) if self.train else None
        encounter_id: str = self.load_encounter_id(qa_pair)
        
        if self.train:
            if self.transforms:
                images = [self.transforms(image=image) for image in images] # at least resizes them
                images  = torch.stack([torch.tensor(image['image']) for image in images])  # list of tensors
            else:
                images  =  torch.stack([torch.tensor(image) for image in images])
            
            if len(images) < len(answers):
                # bootstrap images to the same size as answers
                sampled_img_indices: ndarray = np.random.choice(len(images), len(answers) - len(images))
                #pdb.set_trace()
                sampled_images: torch.Tensor = images[sampled_img_indices]
                sampled_images_augmented: torch.Tensor = torch.vstack([images, sampled_images])
                return (encounter_id, self.language), sampled_images_augmented, question, answers  
            elif len(answers) < len(images):
                # bootstrap answers
                remaining_answers = len(images) - len(answers)

                # Fill the remaining entries with random samples, preferring longer strings
                additional_answers = []
                for _ in range(remaining_answers):
                    if random.random() < self.long_answer_ratio:  # Adjust preference for longer strings as needed
                        sampled_answer = max(answers, key=len)  # Sample from existing answers based on length
                    else:
                        sampled_answer = random.choice(answers)  # Randomly sample from existing answers
                    additional_answers.append(sampled_answer)

                # Combine existing and additional answers
                expanded_answers = answers + additional_answers
                return (encounter_id, self.language), images, question, expanded_answers
            else:
                assert len(images) == len(answers)
                return (encounter_id, self.language), images, question, answers
            
        else:   # validation/testing
            if self.transforms:
                images = [self.transforms(image=image) for image in images]
                images  = torch.stack([torch.tensor(image['image']) for image in images])  # list of tensors
            else:
                images  =  torch.stack([torch.tensor(image) for image in images])

            return (encounter_id, self.language), images, question
        

class BuildVQADatasetMultilingual(torch.utils.data.Dataset):
    '''
    Loading VQA dataset for multiple languages
    Args:
        json_file - str - directory where train.json file can be found
        train_dir - str - directory where images for training, val and testing are found
        transforms - Albumentations - image transformations defined in CFG file
        train - bool - flag indicating whether to load images, Q&A or only images and questions
        long_answer_ratio - float - number indicating preference for longer answers during bootstrapping
        
    Returns:
        encounter_id - str - unique ID for writing into prediction.json file
        images - torch.Tensor - stack of dermatology images
        question - str - unique patient's question
        answers - List[str] - list of answers given by different doctors
        
        encounter_id, images, question, answers if train = True
        encounter_id, images, question if train = False

    '''
    def __init__(self, json_file='/kaggle/input/mediqa-m3g/mediqa-m3-clinicalnlp2024/train.json', 
                 train_dir='/kaggle/input/mediqa-m3g/images_final/images_train', 
                 transforms=None, train=True, long_answer_ratio=0.7):
        with open(json_file) as f:
            self.qa_pairs = json.load(f)
        self.load_encounter_id = lambda qa_pairs: qa_pairs['encounter_id']
        self.transforms = transforms
        self.train_dir = train_dir
        self.train = train
        self.long_answer_ratio = long_answer_ratio
        self.total_patients = len(self.qa_pairs)
        
    def load_image_paths(self, encounter_id):
        return [os.path.join(self.train_dir, img_id) for img_id in encounter_id]
    
    def load_questions(self, qa_pairs, language):
        return qa_pairs[f'query_content_{language}']
    
    def load_answers(self, qa_pairs, language):
        if self.train:
            return [answer[f'content_{language}'].replace('\n', '') for answer in qa_pairs['responses']]
        else:
            return None
    
    def __len__(self):
        return self.total_patients * 3  # English, Chinese, and Spanish
    
    def __getitem__(self, index):
        # Determine the language based on the index
        if index < self.total_patients:
            language = 'en'
        elif index < self.total_patients * 2:
            language = 'zh'
            index -= self.total_patients  # Adjust index for Chinese
        else:
            language = 'es'
            index -= self.total_patients * 2  # Adjust index for Spanish
        
        qa_pair = self.qa_pairs[index]
        img_paths = self.load_image_paths(qa_pair['image_ids'])
        images = [load_img(img_path) for img_path in img_paths]
        question: str = self.load_questions(qa_pair, language)
        answers: List[str] = self.load_answers(qa_pair, language) if self.train else None
        encounter_id: str = self.load_encounter_id(qa_pair)

        if self.train:
            if self.transforms:
                images = [self.transforms(image=image) for image in images] # at least resizes them
                images  = torch.stack([torch.tensor(image['image']) for image in images])  # list of tensors
            else:
                images  =  torch.stack([torch.tensor(image) for image in images])
            
            if len(images) < len(answers):
                # bootstrap images to the same size as answers
                sampled_img_indices: ndarray = np.random.choice(len(images), len(answers) - len(images))
                #pdb.set_trace()
                sampled_images: torch.Tensor = images[sampled_img_indices]
                sampled_images_augmented: torch.Tensor = torch.vstack([images, sampled_images])
                return (encounter_id, f'content_{language}'), sampled_images_augmented, question, answers  
            elif len(answers) < len(images):
                # bootstrap answers
                remaining_answers = len(images) - len(answers)

                # Fill the remaining entries with random samples, preferring longer strings
                additional_answers = []
                for _ in range(remaining_answers):
                    if random.random() < self.long_answer_ratio:  # Adjust preference for longer strings as needed
                        sampled_answer = max(answers, key=len)  # Sample from existing answers based on length
                    else:
                        sampled_answer = random.choice(answers)  # Randomly sample from existing answers
                    additional_answers.append(sampled_answer)

                # Combine existing and additional answers
                expanded_answers = answers + additional_answers
                return (encounter_id, f'content_{language}'), images, question, expanded_answers
            else:
                assert len(images) == len(answers)
                return (encounter_id, f'content_{language}'), images, question, answers
            
        else:   # validation/testing
            if self.transforms:
                images = [self.transforms(image=image) for image in images]
                images  = torch.stack([torch.tensor(image['image']) for image in images])  # list of tensors
            else:
                images  =  torch.stack([torch.tensor(image) for image in images])

            return (encounter_id, f'content_{language}'), images, question
        

#### Precomputed Image Features
class VQADatasetSingleLanguage(torch.utils.data.Dataset):
    '''
    Loading VQA dataset for a single language
    Args:
        json_file - str - directory where train.json file can be found
        train_dir - str - directory where images for training, val and testing are found
        transforms - Albumentations - image transformations defined in CFG file
        train - bool - flag indicating whether to load images, Q&A or only images and questions
        long_answer_ratio - float - number indicating preference for longer answers during bootstrapping
        language - str - indicates which language to load
    Returns:
        encounter_id - str - unique ID for writing into prediction.json file
        images - torch.Tensor - stack of dermatology images
        question - str - unique patient's question
        answers - List[str] - list of answers given by different doctors
        
        encounter_id, images, question, answers if train = True
        encounter_id, images, question if train = False

    '''
    def __init__(self,json_file='/kaggle/input/mediqa-m3g/mediqa-m3-clinicalnlp2024/train.json', train_dir='/kaggle/input/mediqa-m3g/images_final/images_train', 
                 transforms=None, train=True, long_answer_ratio = 0.7, language = 'content_en', feature_extractor='pathgen', split='train'):
        with open(json_file) as f:
            self.qa_pairs = json.load(f)
        self.transforms = transforms
        self.train_dir = train_dir
        self.load_encounter_id = lambda qa_pairs: qa_pairs['encounter_id']
        self.load_image_paths = lambda encounter_id: [os.path.join(self.train_dir, img_id) for img_id in encounter_id]
        self.load_questions = lambda qa_pairs: qa_pairs[f'query_{language}']
        self.train = train
        self.load_answers = lambda qa_pairs: [answer[f'{language}'] for answer in qa_pairs['responses']] if self.train else None
        self.long_answer_ratio = long_answer_ratio # for bootstrapping longer answers
        self.language = language
        self.feature_extractor = feature_extractor
        self.split = split
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, index):
        qa_pair = self.qa_pairs[index]

        img_paths = self.load_image_paths(qa_pair['image_ids'])
        images = [load_img(img_path)  for img_path in img_paths]
        question: str = self.load_questions(qa_pair)
        answers: List[str] = self.load_answers(qa_pair) if self.train else None
        encounter_id: str = self.load_encounter_id(qa_pair)
        feature_dir = os.path.join('Features', self.feature_extractor, self.split)
        if self.train:
            
            if len(images) < len(answers):
                
                img_features = torch.load(os.path.join(feature_dir, f'{encounter_id}.pt'))
                return (encounter_id, self.language), img_features, question, answers  
            elif len(answers) < len(images):
                # bootstrap answers
                remaining_answers = len(images) - len(answers)

                # Fill the remaining entries with random samples, preferring longer strings
                additional_answers = []
                for _ in range(remaining_answers):
                    if random.random() < self.long_answer_ratio:  # Adjust preference for longer strings as needed
                        sampled_answer = max(answers, key=len)  # Sample from existing answers based on length
                    else:
                        sampled_answer = random.choice(answers)  # Randomly sample from existing answers
                    additional_answers.append(sampled_answer)

                # Combine existing and additional answers
                expanded_answers = answers + additional_answers
                img_features = torch.load(os.path.join(feature_dir, f'{encounter_id}.pt'))
                return (encounter_id, self.language), img_features, question, expanded_answers
            else:
                assert len(images) == len(answers)
                img_features = torch.load(os.path.join(feature_dir, f'{encounter_id}.pt'))
                return (encounter_id, self.language), img_features, question, answers
            
        else:   # validation/testing
            img_features = torch.load(os.path.join(feature_dir, f'{encounter_id}.pt'))
            return (encounter_id, self.language), img_features, question

class MedVQADataset(Dataset):
    def __init__(self, data, split='train',like_test=False,prefix_length=2,model_type = 'gpt2',
                 structured_prompt = ["Tú eres un dermatólogo experto y recibes esta pregunta:  ", 
                                      " dado este contexto de imágenes tomadas por el paciente: ",
                                      " Explica y ofrece una respuesta: "]):
        super().__init__()
        # data = VQADatasetSingleLanguage()
        sys.stdout.flush()
        self.data = data
        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_seqs_len = (512, 512) # pad with a max sequence length of 512 for questions and answers data['max_seqs_len']
        # self.labels = data['class_ids']       
        self.train_setting = True if (split!='test'and like_test==False) else False
        self.prefix_len = prefix_length
        self.structured_prompt = structured_prompt
        
    def __len__(self):
        return len(self.data)
    def pad_sequences(self,index):
        ### PROMPT ENGINEERING
        m = [torch.tensor(self.tokenizer.encode(self.structured_prompt[0])), \
             torch.tensor(self.tokenizer.encode(self.structured_prompt[1])),\
             torch.tensor(self.tokenizer.encode(self.structured_prompt[2])),torch.tensor(self.tokenizer.encode('<|endoftext|>'))]
        m_mask = [torch.ones(len(self.tokenizer.encode(self.structured_prompt[0]))),
                  torch.ones(len(self.tokenizer.encode(self.structured_prompt[1]))),
                  torch.ones(len(self.tokenizer.encode(self.structured_prompt[2]))),
                  torch.zeros(self.tokenizer.encode('<|endoftext|>'))]   

        if self.train_setting:
            # construct the model input. The order is question, image, answer. During training the answer is masked. Any padding is placed on the right of the sequence. 
            # placeholder tokens are used on the location where the visual prefix will be inserted, with q_len indicating this location. 
            q=torch.tensor(self.tokenizer.encode(self.data[index][2])) # 1 question
            
            answers = self.data[index][3]
            encodings = self.tokenizer.batch_encode_plus(
                answers,  # Include the question and answers
                # max_length=self.max_seqs_len[1],  # Set a max length for padding
                padding='longest',  # Pad to the longest sequence in the batch
                truncation=True,  # Truncate sequences longer than max_length
                return_tensors='pt'  # Return PyTorch tensors
            )
            # a=torch.tensor(self.tokenizer.encode(str(self.data[index][3]))) # multiple answers
            # 
            a = torch.tensor(encodings['input_ids'], dtype=torch.float16)
            
            # padding the question
            q,q_mask,_ = self.make_padding(self.max_seqs_len[0],q,question=True) # leftover tokens is max seq length like 512 - questions tokens such as 190 in this case
            q_len = m[0].size(0) + q.size(0) + m[1].size(0) # pregunta's tokens + question's token + contexto
            
            # padding and masking the answer
            # a,a_mask,_ = self.make_padding(self.max_seqs_len[1],a,leftover_tokens=leftover_tokens)
            # a,a_mask= self.make_2d_padding_for_answer(self.max_seqs_len[1], a, leftover_tokens=leftover_tokens)
            # if len((a==0).nonzero())!=0:
            #     pad_start = (a==0).nonzero()[0] # returns list of indices where tokens are zero in the answer string. Indices are 2D if answers is 2D.
            # else:
            #     pad_start=[]
            # pdb.set_trace()
            pad_start = []
            for row_answer in range(a.size(0)):
                if len((a[row_answer] == 0).nonzero()) != 0:
                    pad_start.append((a[row_answer]==0).nonzero()[0].item())
            a_padded = np.zeros((a.shape[0], a.shape[1] + 1), dtype=np.int16) # +1 for the m[3] <endoftext> token
            # pdb.set_trace()
            for i in range(a.size(0)):  # Loop through each answer
                # if len(pad_start) == 0:  # No padding found or exceeds size
                    a_padded[i] = torch.cat((a[i], m[3]))  # Append special token at end
                # else:
                    # pdb.set_trace()
                #    a_padded[i] = torch.cat((a[i][:pad_start[i]], m[3], a[i][pad_start[i]:]))  # Insert special token at padding location
                    # pdb.set_trace()
            num_rows = a.shape[0]
            # a = torch.cat((a,m[3])) if len(pad_start)==0 else torch.cat((a[:pad_start],m[3],a[pad_start:]))
            a_padded = torch.from_numpy(a_padded)
            
            q = torch.cat((m[0].repeat(num_rows, 1), # "question: "
                           q.repeat(num_rows, 1),    # the question                   
                           m[1].repeat(num_rows, 1), # "context: "
                           torch.ones(num_rows, self.prefix_len), # image embeddings
                           m[2].repeat(num_rows, 1), # "answers: "
                           a_padded),                 # answer and <endoftext> special token
                           dim = 1) # updated a variable q which is the structured query: question: question token, context: image embeddings, answer: answer tokens, <endoftext>
            # pdb.set_trace()
            # q_mask = torch.cat((m_mask[0],q_mask,m_mask[1],torch.ones(self.prefix_len),m_mask[2],a_mask,m_mask[3]))
            q_mask = torch.cat((m_mask[0].repeat(num_rows, 1),
                                q_mask.repeat(num_rows, 1),
                                m_mask[1].repeat(num_rows, 1),
                                torch.ones(num_rows, self.prefix_len),
                                m_mask[2].repeat(num_rows, 1),
                                torch.zeros_like(a_padded)), 
                                dim=1)
            
            # self.tokenizer.batch_decode(torch.tensor(q, dtype=torch.int32))
            # pdb.set_trace()

            return q,q_mask, q_len
        else:
            # in the test stage we do not have acces to the answer, so we just load the question. 
            # since inference is not performed batch-wised we don't need to apply padding
            q = torch.tensor(self.tokenizer.encode(self.data[index][2]))
            num_answers = self.data[index][1].shape[0] # this is prefix's index
            q,q_mask,_ = self.make_padding_test_setting(self.max_seqs_len[0],q)
            q_len = m[0].size(0) + q.size(0) + m[1].size(0)
            # pdb.set_trace()
            q = torch.cat((m[0].repeat(num_answers, 1),
                           q.repeat(num_answers, 1),
                           m[1].repeat(num_answers, 1),
                           torch.ones(num_answers, self.prefix_len),
                           m[2].repeat(num_answers, 1)), dim = 1)
            
            
            q_mask = torch.cat((m_mask[0].repeat(num_answers, 1),
                                q_mask.repeat(num_answers, 1),
                                m_mask[1].repeat(num_answers, 1),
                                torch.ones(num_answers, self.prefix_len),
                                m_mask[2].repeat(num_answers, 1)), dim = 1) # concatenating tokens question:1mask, context: vvv, answer: no answer since it's inference, 
            return q, q_mask, q_len 

    def make_padding(self, max_len, tokens, question=False,leftover_tokens=0):
        padding = max_len - tokens.size(0) 
        if padding > 0:
            if question:
                leftover_tokens = padding
                mask = torch.ones(tokens.size(0)) # if tokens is less than max_len, mask is same size of tokens
            else:
                tokens = torch.cat((tokens, torch.zeros(padding+leftover_tokens))) # for answers, 
                mask = torch.zeros(max_len+leftover_tokens)    
              
        elif padding==0:
            if question:
                mask = torch.ones(tokens.size(0)) 
            else:
                mask = torch.zeros(tokens.size(0)+leftover_tokens)
                tokens = torch.cat((tokens,torch.zeros(leftover_tokens)))
                
        
        elif padding < 0:
            if question:
                tokens = tokens[:max_len]
                mask = torch.ones(max_len)
            else:
                tokens = torch.cat((tokens[:max_len], torch.zeros(leftover_tokens)))
                mask = torch.zeros(max_len+ leftover_tokens)
        return tokens, mask, leftover_tokens
    
    def make_2d_padding_for_answer(self, max_len, tokens, leftover_tokens):
        # This function assumes tokens is a 2D tensor where each row corresponds to an answer
        # Prepare an empty list to hold padded tokens and masks for each answer
        padded_tokens = []
        masks = []

        for t in tokens:
            padding = max_len - t.size(0)  # Calculate padding for the current answer
            if padding > 0:
                # Pad with zeros
                # pdb.set_trace()
                padded_t = torch.cat((t, torch.zeros(padding)))  # torch.cat((t, torch.zeros(padding)))  
                mask =  torch.zeros(max_len)  # torch.ones(t.size(0)).tolist() + [0] * padding  # Mask 1s for valid tokens, 0s for padding
            elif padding == 0:
                padded_t = torch.cat((t,torch.zeros(leftover_tokens))) # t
                mask = torch.zeros(t.size(0)+leftover_tokens) # torch.ones(t.size(0)).tolist()  # No padding, all tokens are valid
            else:
                # Truncate the tokens if they exceed max_len
                padded_t = t[:max_len] # torch.cat(t[:max_len]) # t[:max_len]
                mask = torch.zeros(max_len) # torch.ones(max_len).tolist()  # Everything in the truncated part is valid

            padded_tokens.append(padded_t)  # Append the padded tokens
            masks.append(mask)  # Append the mask
        # pdb.set_trace()
        # Convert the list of padded tokens to a tensor
        return torch.stack(padded_tokens), torch.stack(masks)  
    
    def make_padding_test_setting(self, max_len, tokens,do_padding=False):
        padding = max_len - tokens.size(0)
        padding_len = 0
        if padding > 0:
            if do_padding:
                mask = torch.cat((torch.ones(tokens.size(0)),torch.zeros(padding)))
                tokens = torch.cat((tokens,torch.zeros(padding)))
                padding_len = padding
            else:
                mask = torch.ones(tokens.size(0))
        elif padding ==0:
            mask = torch.ones(max_len)
        elif padding < 0:
            tokens = tokens[:max_len]
            mask = torch.ones(max_len)
        return tokens, mask, padding_len
            
    def __getitem__(self, index):
        prefix = self.data[index][1]
        tokens, mask, q_len  = self.pad_sequences(index)
        return prefix, tokens, mask, q_len # tokens are the questions, mask are placed over the answer
    

def get_dataloaders(CFG = CFG):

    train_dataset_es = VQADatasetSingleLanguage(transforms=CFG.data_transforms['train'], 
                                    json_file=CFG.train_json,
                                    train_dir=CFG.train_groups,
                                    language = 'content_es', split='train')


    # val_dir = CFG.valid_groups

    # val_img_paths = list(map(lambda x: os.path.join(val_dir, x), os.listdir(val_dir)))
    val_dataset_es = VQADatasetSingleLanguage(transforms=CFG.data_transforms['valid'], 
                                    json_file=CFG.valid_json,
                                    train_dir=CFG.valid_groups, train=False,
                                    language='content_es', split='valid')


    # test_dir = CFG.test_groups
    test_dataset_es = VQADatasetSingleLanguage(transforms=CFG.data_transforms['valid'], 
                                    json_file=CFG.test_json,
                                    train_dir=CFG.test_groups, train=False,
                                    language='content_es', split='test')

    trainset = MedVQADataset(data=train_dataset_es, prefix_length=8)
    validset = MedVQADataset(data=val_dataset_es, split='valid', like_test=True,  prefix_length=8)
    testset = MedVQADataset(data=test_dataset_es, split='test', like_test=True,  prefix_length=8)

    trainloader = DataLoader(trainset,
                              # collate_fn=custom_collate_fn,
                              batch_size=CFG.train_bs,
                              shuffle=True)

    validloader = DataLoader(validset,
                            # collate_fn=collate_fn,
                            batch_size=CFG.valid_bs,
                            shuffle=False)

    testloader = DataLoader(testset,
                             batch_size=CFG.valid_bs,
                             shuffle=False)
    
    return trainloader, validloader, testloader