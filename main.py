


from args import get_args
from data import get_dataloaders # VQADatasetSingleLanguage, MedVQADataset
from models import VQAmedModel
from CFG import CFG
from train import train
from inference import society_inference

import torch
import pdb


def main(args):
    ### 1. LOAD DATASET AND DATALOADERS
    
    trainloader, validloader, testloader = get_dataloaders(CFG=CFG)

    ### 2. LOAD MODELS
    model = VQAmedModel(
            prefix_length=8, # same as prefix length in MedVQADataset
            clip_length=4,
            setting='lora',
            mapping_type="MLP", 
            # default model type is gpt2-xl
        )
    model2 = VQAmedModel(
            prefix_length=8, # same as prefix length in MedVQADataset
            clip_length=4,
            setting='lora',
            mapping_type="MLP",
            model_type="microsoft/biogpt"
        )
    # model3 = VQAmedModel(
    #         prefix_length=8, # same as prefix length in MedVQADataset
    #         clip_length=4,
    #         setting='lora',
    #         mapping_type="MLP",
    #         model_type="stanford-crfm/BioMedLM"
    #     )
    
    ensemble = [model, model2]
    
    ### 3. TRAIN MODELS
    if args.split == "train":

        for idx in range(len(ensemble)):
            ensemble[idx] = train(trainloader, validloader, ensemble[idx], args)
            torch.save(ensemble[idx].state_dict(), f'model_{idx}.pth')
    
    ### 4. PERFORM INFERENCE
    elif args.split == "test":
        ### Write code to load the model weights
        # ensemble[idx].load_state_dict(torch.load(model_weights_dir))
        ###
        val_score = society_inference(dataloader=testloader, ensemble=ensemble, args=args, reference_dir='mediqa-m3g-startingkit-v2/app/input/ref/reference_test.json')
        test_score = society_inference(dataloader=testloader, ensemble=ensemble, args=args, reference_dir='m3g-test-ref/reference.json')

        print(val_score, test_score)
if __name__ == "__main__":
    args = get_args()
    main(args)

    