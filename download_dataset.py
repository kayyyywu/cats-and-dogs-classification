from torchvision.datasets import OxfordIIITPet
import os
import pandas as pd
from tqdm import tqdm
import pickle
from my_utils import clip_preprocess, PET_CLASSES


# Download to folder "CLIP"
def download_pet_images(directory=r"datasets", img_transform = clip_preprocess):
    if os.path.isdir(directory + "IIIT-pets"):
        do_download = False
    else:
        do_download = True
    train_set = OxfordIIITPet(root=directory, split='trainval', transform=img_transform, download=do_download)
    test_set = OxfordIIITPet(root=directory, split='test', transform=img_transform, download=do_download)

    # Cache data to a file
    with open('datasets/breed_list.pkl', 'wb') as file:
        pickle.dump(train_set.classes, file)

    return train_set, test_set

# Get pet classes
def get_pet_classes():
    if len(PET_CLASSES) == 0:
        breed_list = download_pet_images()[0].classes
    else:
        breed_list = PET_CLASSES
    return breed_list
    
# Read labels
def get_labels(directory=r"datasets", is_train=True):
    if is_train:
        file = "trainval.txt"
    else:
        file = "test.txt"
    curr_directory = os.path.join(directory, "oxford-iiit-pet", "annotations", file)
    df = pd.read_csv(curr_directory, sep=' ', header=None, names=['name_id', 'Class', 'Species', 'Breed'])
  
    return df

# Make Class start with 0
def update_labels(pet_dataset, labels_df):
    updated_dataset = []
    # Use tqdm to create a progress bar
    for n in tqdm(range(len(pet_dataset)), desc="Updating Labels"):
        # change it to start with 0
        updated_dataset.append((pet_dataset[n][0], labels_df['Class'][n] - 1))

    return updated_dataset
