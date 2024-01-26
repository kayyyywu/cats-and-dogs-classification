# cats-and-dogs-classification

## **Developers:**

**Kexian Wu**: App development, YOLO object detection, Clip-zero shot, Clip-linear probe.

**Jialong Li**: Model training, Clip-zero shot, Clip-linear probe, Fine-tuned ResNet.  

## **The link of the APP:**

<https://cats-and-dogs-classification.streamlit.app>

## **Brief Desciption:**

Dogs and cats are the most popular pets in the United States. More than just satisfying curiosity, understanding your pet's breed can provide valuable insights into their behavior.

In the pet breed recognition project, we compared the accuracy of OpenAI **CLIP** zero-shot recognition based on **ResNet** and **ViT** encoder, and ViT wins in all cases. Then training a **linear probe CLIP** boosted accuracy from 80% to 89%. Additionally, we **Fine-tuned** a **ResNet50** model which further improved accuracy to 93%.

In the app, we employed **YOLOv5** to remove the background from images and resize them into squares while preserving the aspect ratio through padding to match the input size required by the Convolutional Neural Network. YOLOv5 can also detect whether the input image contains cats and dogs, ensuring that the model avoids generating inappropriate responses. Unfortunately, due to RAM constraints, we couldn't include the fine-tuned ResNet50 model in the app.
