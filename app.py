import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import json
import os
# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="Food Image Classifier",
    layout="centered"
)

MODEL_DIR = "export"   # folder where model_scripted.pt is stored
TORCHSCRIPT_PATH = os.path.join(MODEL_DIR, "model_scripted.pt")
STATE_DICT_PATH = os.path.join(MODEL_DIR, "best_model.pth")
CLASSES = [
    "cannoli", "ceviche", "crab_cakes", "frozen_yogurt", "gnocchi",
    "grilled_cheese_sandwich", "onion_rings", "pork_chop",
    "ravioli", "spaghetti_bolognese"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# LOAD MODEL (TorchScript first)
# -------------------------------
@st.cache_resource
def load_model():
    if os.path.exists(TORCHSCRIPT_PATH):
        st.success("Loaded TorchScript model 🎉")
        model = torch.jit.load(TORCHSCRIPT_PATH, map_location=device)
        model.eval()
        return model
    
    else:
        st.warning("TorchScript not found — loading state_dict instead.")

        model = timm.create_model("resnet101", pretrained=False, num_classes=len(CLASSES))
        state = torch.load(STATE_DICT_PATH, map_location=device)
        model.load_state_dict(state)
        model = model.to(device).eval()
        return model


model = load_model()

# -------------------------------
# IMAGE TRANSFORMS
transform=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict(img):
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_t)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    top3_idx=probs.argsort()[-3:][::-1]
    top3=[(CLASSES[i], float(probs[i])) for i in top3_idx]
    return top3, probs

#Streamlit app's UI
st.title("Food Image Classifier")
st.write("Upload a food image and let the model classify it!")
uploaded=st.file_uploader("Upload an image",type=["jpg","jpeg","png"])
if uploaded:
    img=Image.open(uploaded).convert("RGB")
    st.image(img,caption="Uploaded Image",use_column_width=True)
    st.write("**Classifying...**")
    top3,full_probs=predict(img)
    st.subheader("Top Prediction")
    st.write(f"**{top3[0][0]}** — {top3[0][1]*100:.2f}% confidence")

    st.subheader(" Other likely predictions")
    for cls, prob in top3[1:]:
        st.write(f"- **{cls}** — {prob*100:.2f}%")

    #Show probabilities per class (this is completely optional)
    with st.expander("See full probability distribution"):
        prob_dict={CLASSES[i]: float(full_probs[i]) for i in range(len(CLASSES))}
        st.json(prob_dict)
st.write("---")
st.caption("Built using PyTorch, Timm & Streamlit")