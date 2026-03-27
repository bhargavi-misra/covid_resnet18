import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import requests
import io
import os

# -------- CONFIG --------
MODEL_URL = "https://huggingface.co/bhargavi-misra/covidresnet18/resolve/main/covid_resnet18_fast.pth"
CLASSES = ["Normal", "COVID"]

# -------- LOAD MODEL FROM URL --------
@st.cache_resource
def load_model():
    with st.spinner("Fetching model from Hugging Face... ⏳"):
        response = requests.get(MODEL_URL)

        if response.status_code != 200:
            st.error("Failed to download model")
            st.stop()

        model_state_dict = torch.load(io.BytesIO(response.content), map_location="cpu")

    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(model_state_dict)
    model.eval()

    return model

model = load_model()

# -------- PREPROCESS --------
def convert_to_rgb(img):
    return img.convert("RGB")

transform = transforms.Compose([
    transforms.Lambda(convert_to_rgb),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------- UI --------
st.title("🧠 COVID Detection using ResNet18")

mode = st.radio("Choose input method:", ["Sample Images", "Upload Image"])

image = None

# -------- OPTION 1: SAMPLE IMAGES --------
if mode == "Sample Images":
    image_options = {
        "COVID Sample 1": "images/Covid1.jpeg",
        "COVID Sample 2": "images/Covid2.jpeg",
        "Normal Sample": "images/Non-Covid1.jpeg"
    }

    selected_image = st.selectbox("Choose an image:", list(image_options.keys()))
    img_path = image_options[selected_image]

    if not os.path.exists(img_path):
        st.error(f"Image not found: {img_path}")
        st.stop()

    try:
        image = Image.open(img_path)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()

# -------- OPTION 2: UPLOAD IMAGE --------
else:
    uploaded_file = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Error loading uploaded image: {e}")
            st.stop()

# -------- DISPLAY IMAGE --------
if image is not None:
    st.image(image, caption="Selected Image", use_container_width=True)

    # -------- PREDICTION --------
    if st.button("Predict"):
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            pred = torch.argmax(probs).item()

        st.success(f"Prediction: {CLASSES[pred]}")
        st.write(f"Confidence: {probs[pred]*100:.2f}%")