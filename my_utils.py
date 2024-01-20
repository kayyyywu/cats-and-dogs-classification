import torch
import clip
import xml.etree.ElementTree as ET
from PIL import Image


# Check if CUDA (GPU) is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the CLIP model and preprocessing pipeline
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

target_class_ids = [15, 16, 17, 18, 19, 20, 21, 22]
YOLO5_CLASSES = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
            'scissors', 'teddy bear', 'hair drier', 'toothbrush' ]

PET_CLASSES = ['Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair', 'Chihuahua', 'Egyptian Mau', 'English Cocker Spaniel', 'English Setter', 'German Shorthaired', 'Great Pyrenees', 'Havanese', 'Japanese Chin', 'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard', 'Samoyed', 'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier', 'Wheaten Terrier', 'Yorkshire Terrier', 'Ocicat', 'Himalayan', 'Burmese', 'Tortoiseshell', 'Selkirk Rex', 'American Shorthair', 'Siberian', 'Balinese', 'Dilute Calico', 'Turkish Van', 'American Bobtail', 'Manx', 'Hemingway Polydactyl', 'Cornish Rex', 'Devon Rex', 'Korat', 'Silver', 'Tiger', 'Torbie', 'Scottish Fold', 'Turkish Angora', 'Snowshoe', 'Tuxedo', 'Nebelung', 'Oriental Short Hair', 'Tonkinese', 'Ragamuffin', 'Havana', 'Dilute Tortoiseshell', 'Domestic Medium Hair', 'Domestic Short Hair', 'American Curl', 'Applehead Siamese', 'Munchkin', 'Oriental Tabby', 'Calico', 'Chartreux', 'Exotic Shorthair', 'Domestic Long Hair', 'Pixiebob', 'Tabby', 'Japanese Bobtail', 'Norwegian Forest Cat', 'Silky Terrier', 'Scottish Deerhound', 'Chesapeake Bay Retriever', 'Ibizan Hound', 'Wire Haired Fox Terrier', 'Saluki', 'Cocker Spaniel', 'Schipperke', 'Borzoi', 'Pembroke', 'Komondor', 'Standard Poodle', 'Eskimo Dog', 'English Foxhound', 'Golden Retriever', 'Sealyham Terrier', 'Japanese Spaniel', 'Miniature Schnauzer', 'Malamute', 'Malinois', 'Pekinese', 'Giant Schnauzer', 'Mexican Hairless', 'Doberman', 'Standard Schnauzer', 'Dhole', 'German Shepherd', 'Bouvier Des Flandres', 'Siberian Husky', 'Norwich Terrier', 'Irish Terrier', 'Norfolk Terrier', 'Border Terrier', 'Briard', 'Tibetan Mastiff', 'Bull Mastiff', 'Maltese Dog', 'Kerry Blue Terrier', 'Kuvasz', 'Greater Swiss Mountain Dog', 'Lakeland Terrier', 'Blenheim Spaniel', 'Basset', 'West Highland White Terrier', 'Border Collie', 'Redbone', 'Irish Wolfhound', 'Bluetick', 'Miniature Poodle', 'Cardigan', 'Entlebucher', 'Norwegian Elkhound', 'German Short Haired Pointer', 'Bernese Mountain Dog', 'Papillon', 'Tibetan Terrier', 'Gordon Setter', 'American Staffordshire Terrier', 'Vizsla', 'Kelpie', 'Weimaraner', 'Chow', 'Old English Sheepdog', 'Rhodesian Ridgeback', 'Shih Tzu', 'Affenpinscher', 'Whippet', 'Sussex Spaniel', 'Otterhound', 'Flat Coated Retriever', 'Italian Greyhound', 'Labrador Retriever', 'Collie', 'Cairn', 'Rottweiler', 'Australian Terrier', 'Toy Terrier', 'Shetland Sheepdog', 'African Hunting Dog', 'Walker Hound', 'Lhasa', 'Great Dane', 'Airedale', 'Bloodhound', 'Irish Setter', 'Dandie Dinmont', 'Basenji', 'Bedlington Terrier', 'Appenzeller', 'Clumber', 'Toy Poodle', 'English Springer', 'Afghan Hound', 'Brittany Spaniel', 'Welsh Springer Spaniel', 'Boston Bull', 'Dingo', 'Soft Coated Wheaten Terrier', 'Curly Coated Retriever', 'French Bulldog', 'Irish Water Spaniel', 'Brabancon Griffon', 'Groenendael', 'Black And Tan Coonhound']


def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im
