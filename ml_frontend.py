import streamlit as st
from PIL import Image, ImageOps
from io import BytesIO
import pandas as pd
import torch
from model import clip_zero_shot, linear_probe, get_preds
from download_dataset import get_pet_classes
from my_utils import device, clip_preprocess, target_class_ids, make_square
import numpy as np
import cv2
from streamlit_extras.buy_me_a_coffee import button


# set device to "cuda" to call the GPU
device = "cuda" if torch.cuda.is_available() else 'cpu'
# set page layout
st.set_page_config(
    page_title="What Breed is This Dog or Cat?",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)
button(username="cats_and_dogs", floating=True, width=221)
st.title("What Breed is This Dog or Cat?")
st.sidebar.subheader("Input")
models_list = ["Linear Probe", "Zero-shot"]
selected_model = st.sidebar.selectbox("Select the Model", models_list)

# component to upload images
uploaded_file = st.sidebar.file_uploader(
    "Choose an image to classify", type=["jpg", "jpeg", "png"]
)
st.subheader('How to choose the right model :sunglasses:')
st.info('''
        Zero-shot: Stable performance across different image qualities.
        
        Linear probe: Superior for high-quality images with a square-like shape, where the pet occupies a significant portion of the frame.
        ''')

if uploaded_file:
    bytes_data = uploaded_file.read()
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = get_preds(img)

    result_copy = result.copy()
    if len(result_copy) > 0 and np.any(np.isin(result_copy[:,-1], target_class_ids)):
        img_draw = img.copy().astype(np.uint8)
        bbox_data = result_copy[np.isin(result_copy[:,-1], target_class_ids)][0]
        xmin, ymin, xmax, ymax, _, label = bbox_data
        p0, p1, label = (int(xmin), int(ymin)), (int(xmax), int(ymax)), int(label)
        img_draw = cv2.rectangle(img_draw, 
                                p0, p1, 
                                (255, 0, 0), 2) 
        st.header('Before Crop Adjustment', divider='rainbow')
        st.image(img_draw, caption=["Uncropped Image"])
        cropped_image = img[p0[1]:p1[1], :].copy()
        cropped_image = Image.fromarray(cropped_image)

    image_input = clip_preprocess(cropped_image).unsqueeze(0).to(device)

    if selected_model == "Zero-shot":
        values, indices = clip_zero_shot(image_input, k=5)
    else:
        values, indices = linear_probe(image_input, k=5)

    if not np.any(np.isin(result_copy[:,-1], target_class_ids)) or max(values) < 0.23:
        st.image(bytes_data,
            caption=[f"Original Image"],
        )
        st.error("Apologies, couldn't identify the breed of the uploaded image at this time..", icon="🚨")
    else:
        pet_classes = get_pet_classes()
        st.header('After Crop Adjustment', divider='rainbow')
        st.image(cropped_image, 
            caption=[f"{pet_classes[indices[0]]} {values[0]*100:.2f}%"],
        )
        st.subheader(f"Top Predictions from {selected_model}")

        # Create a list of dictionaries for each prediction
        predictions_list = [
            {"Classification": pet_classes[index], "Confidence": "{:.2f} %".format(100 * value)}
            for value, index in zip(values, indices)
        ]
        st.dataframe(
            pd.DataFrame(predictions_list)
        )
