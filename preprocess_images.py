# import requests
import os
import cv2
import torch
import open_clip
from CFG import CFG
from data import BuildVQADatasetSingleLanguage

def extract_image_features(dataset, model, split, feature_dir, recompute=False):
    """
    Extract image features using a pre-trained model and save them to disk,
    with an option to recompute existing features.
    
    Args:
        dataset (torch.utils.data.Dataset): The dataset containing the images.
        model (torch.nn.Module): The pre-trained model used for feature extraction.
        split (str): The dataset split ('train', 'valid', or 'test').
        recompute (bool): If True, recompute features even if files already exist.
    """
    # feature_dir = "Features/pathgen"
    os.makedirs(f'{feature_dir}/{split}', exist_ok=True)
    
    for i in range(len(dataset)):
        img_id = dataset[i][0][0]
        output_file = os.path.join(feature_dir, split, f'{img_id}.pt')
        
        # Check if the file already exists and recompute flag
        if recompute or not os.path.exists(output_file):
            with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                image = dataset[i][1]
                image = image.permute(0, 3, 1, 2)
                image_features = model.encode_image(image)
                torch.save(image_features, output_file)

if __name__ == "__main__":
    # 1. Load dataset
    train_dataset_es = BuildVQADatasetSingleLanguage(transforms=CFG.data_transforms['train'], 
                                json_file=CFG.train_json,
                                train_dir=CFG.train_groups,
                                language = 'content_es')


    val_dir = CFG.valid_groups

    val_img_paths = list(map(lambda x: os.path.join(val_dir, x), os.listdir(val_dir)))
    val_dataset_es = BuildVQADatasetSingleLanguage(transforms=CFG.data_transforms['valid'], 
                                    json_file=CFG.valid_json,
                                    train_dir=CFG.valid_groups, train=False,
                                    language='content_es')


    test_dir = CFG.test_groups
    test_dataset_es = BuildVQADatasetSingleLanguage(transforms=CFG.data_transforms['valid'], 
                                    json_file=CFG.test_json,
                                    train_dir=CFG.test_groups, train=False,
                                    language='content_es')
    # 2. Load pretrained image encoder 
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16')
    model.load_state_dict(torch.load('pathgenclip.pt'))
    # pathgenclip.pt is in HugginFace https://github.com/mlfoundations/open_clip?tab=readme-ov-file
    print("All keys matched succesfully")
    model.eval()
    print("PathGen loaded succesfully")

    # 3. Compute and save image features
    feature_extractor = 'pathgen'
    feature_dir = os.path.join('Features', feature_extractor)
    extract_image_features(dataset=train_dataset_es, model=model, split='valid', feature_dir=feature_dir, recompute=False)
    extract_image_features(dataset=val_dataset_es, model=model, split='valid', feature_dir=feature_dir, recompute=False)
    extract_image_features(dataset=test_dataset_es, model=model, split='test', feature_dir=feature_dir, recompute=False)