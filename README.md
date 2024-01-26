# cats-and-dogs-classification

## **Developers:**

**Kexian Wu**: App development, YOLO object detection, Clip-zero shot, Clip-linear probe.

**Jialong Li**: Model training, Clip-zero shot, Clip-linear probe, Fine-tuned ResNet.  

## **The link of the APP:**

<https://cats-and-dogs-classification.streamlit.app>

## **Project desciption:**

Dogs and cats are the most popular pets in the United States. More than just satisfying curiosity, understanding your pet's breed can provide valuable insights into their behavior.

In the pet breed recognition project, we compared the accuracy of OpenAI **CLIP** zero-shot recognition based on **ResNet** and **ViT** encoder, and ViT wins in all cases. Then training a **linear probe CLIP** boosted accuracy from 80% to 89%. Additionally, we **Fine-tuned** a **ResNet50** model which further improved accuracy to 93%.

In the app, we employed **YOLOv5** to remove the background from images and resize them into squares while preserving the aspect ratio through padding to match the input size required by the Convolutional Neural Network. YOLOv5 can also detect whether the input image contains cats and dogs, ensuring that the model avoids generating inappropriate responses. Unfortunately, due to RAM constraints, we couldn't include the fine-tuned ResNet50 model in the app.

We chose Oxford pets[1] as our first training dataset, and further increased the number of breeds from 37 to 184 with the kaggle cat breeds dataset[2] and Stanford dogs dataset[3]. Currently the app has achieve a high perfomance on dog breed classification, but relatively low performance on cat breed classification due to the low quality of the kaggle cat breeds dataset. We are actively seeking high quality cat breeds dataset to upgrade our app.

We selected Oxford Pets[1] as our initial training dataset and expanded the breed diversity from 37 to 184 by incorporating the Kaggle Cat Breeds dataset[2] and the Stanford Dogs dataset[3]. While our application excels in dog breed classification, it currently hsa relatively lower accuracy in cat breed classification. This can be attributed to the low quality of the Kaggle Cat Breeds dataset. We are actively seeking a better cat breeds dataset to enhance the overall performance of our application.

## **Datasets:**
[1] https://www.robots.ox.ac.uk/~vgg/data/pets/
[2] https://www.kaggle.com/datasets/ma7555/cat-breeds-dataset/data
[3] http://vision.stanford.edu/aditya86/ImageNetDogs/
