import model
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
import albumentations as A
import streamlit as st
import inference

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
device="cuda" if torch.cuda.is_available() else "cpu"


# Streamlit UI
st.title("Image Segmentation App")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    # Display uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Perform segmentation when a button is clicked
    if st.button("Segment Image"):
        # Read the uploaded image as a byte stream
        image_stream = uploaded_image.read()

        # Convert the byte stream to a NumPy array
        np_image = np.frombuffer(image_stream, np.uint8)

        # Decode the NumPy array into an OpenCV image (BGR format)
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        aug=inference.t_test(image=img)
        img=Image.fromarray(aug['image'])

        t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

        image = t(img)
        model.model.to(device)
        image=image.to(device)

        with torch.no_grad():
            
            image = image.unsqueeze(0)
            
            output = model.model(image)
            masked = torch.argmax(output, dim=1)
            masked = masked.cpu().squeeze(0)

            # Display the image using Matplotlib
            fig, ax = plt.subplots()
            ax.imshow(masked)  # Replace 'gray' with the desired colormap
            ax.axis('off')  # Hide the axes
            st.pyplot(fig)  # Display the Matplotlib figure in Streamlit
