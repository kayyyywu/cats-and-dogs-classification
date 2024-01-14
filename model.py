import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import clip
from download_dataset import download_pet_images, update_labels, get_labels, get_pet_classes
from sklearn.linear_model import LogisticRegression
import pickle
from my_utils import device, clip_model, clip_preprocess


def clip_zero_shot(image_input, k=5):
    # get all pet classes
    pet_classes = get_pet_classes()
    # put text to match to image in device memory
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}, a type of pet.") for c in pet_classes]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input) # compute image features with CLIP model
        text_features = clip_model.encode_text(text_inputs) # compute text features with CLIP model
    image_features /= image_features.norm(dim=-1, keepdim=True) # unit-normalize image features
    text_features /= text_features.norm(dim=-1, keepdim=True) # unit-normalize text features

    # Pick the top 5 most similar labels for the image
    similarity = (100.0 * image_features @ text_features.T) # score is cosine similarity times 100
    p_class_given_image = similarity.softmax(dim=-1)  # P(y|x) is score through softmax
    values, indices = p_class_given_image[0].topk(k) # gets the top 5 labels

    return values, indices

def get_features(data_set, encoder = clip_model):
    data_loader = DataLoader(data_set, batch_size=64, shuffle=False)  # dataloader lets you process in batch which is way faster
    image_features = []
    labels = []

    # Extract CLIP features for each image in the dataset
    for images, labels_batch in data_loader:
      # Move images to the device where the model is located
      images = images.to(device)

      # Extract CLIP features for the images
      with torch.no_grad():
        features = encoder.encode_image(images)

      # Append the features and labels to the lists
      image_features.append(features.cpu().numpy())
      labels.append(labels_batch.numpy())

   # Concatenate the features and labels into numpy arrays
    image_features = np.concatenate(image_features)
    labels = np.concatenate(labels)

    return image_features, labels

def train_logistic_regression(transform=clip_preprocess):
    pet_train_trans, _ = download_pet_images(img_transform=transform)
    labels_df = get_labels(is_train=True)
    pet_train_final = update_labels(pet_train_trans, labels_df)
    train_features, train_labels = get_features(pet_train_final)
    clf = LogisticRegression(random_state=0,  max_iter=500)
    clf.fit(train_features, train_labels)
    # Save the model to a file
    folder_path = "model"
    if not os.path.exists(folder_path):
    # If it doesn't exist, create the folder
        os.makedirs(folder_path)

    model_file_name = 'clip_linear_probe.pkl'
    # Combine the folder path and model file name
    model_path = os.path.join(folder_path, model_file_name)
    with open(model_path, 'wb') as file:
        pickle.dump(clf, file)

    return clf

def linear_probe(image_input, k=5):
    # Check if the file exists
    model_path = 'model/clip_linear_probe.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            clf = pickle.load(file)
    else:
        clf = train_logistic_regression()
    # Calculate features
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input) # compute image features with CLIP model
    y_probabilities = clf.predict_proba(image_features)
    indices = np.argsort(y_probabilities[0])[::-1][:k]
    values = y_probabilities[0][indices].tolist()
    
    return values, indices

def get_yolo5(model_type='s'):
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    return torch.hub.load('ultralytics/yolov5', 
                          'yolov5{}'.format(model_type), 
                          pretrained=True
                          )

def get_preds(img):
    model = get_yolo5()
    return model([img]).xyxy[0].numpy()
