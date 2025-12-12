# app.py
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import os

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="Food Image Classifier",
    layout="centered"
)

MODEL_DIR = "export"   # folder where model_scripted.pt or best_model.pth is stored
TORCHSCRIPT_PATH = os.path.join(MODEL_DIR, "model_scripted.pt")
STATE_DICT_PATH = os.path.join(MODEL_DIR, "best_model.pth")
CLASSES = [
    "cannoli", "ceviche", "crab_cakes", "frozen_yogurt", "gnocchi",
    "grilled_cheese_sandwich", "onion_rings", "pork_chop",
    "ravioli", "spaghetti_bolognese"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# LAZY MODEL LOADER (cached)
# -------------------------------
@st.cache_resource
def load_model_cached():
    """
    Load and return a model. No Streamlit UI calls inside this function.
    Returns: (model, source_str) where source_str explains what was loaded.
    """
    # Try TorchScript first
    if os.path.exists(TORCHSCRIPT_PATH):
        model = torch.jit.load(TORCHSCRIPT_PATH, map_location=device)
        model.eval()
        return model, f"TorchScript ({TORCHSCRIPT_PATH})"

    # Fall back to state_dict
    if os.path.exists(STATE_DICT_PATH):
        model = timm.create_model("resnet101", pretrained=False, num_classes=len(CLASSES))
        state = torch.load(STATE_DICT_PATH, map_location=device)
        # If your state dict was saved as {'model': state_dict} you'll need to adapt:
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model = model.to(device).eval()
        return model, f"State dict ({STATE_DICT_PATH})"

    # If neither exists, return None
    return None, "Model files not found"

# -------------------------------
# IMAGE TRANSFORMS
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# PREDICTION (lazy-load inside)
# -------------------------------
def predict(img):
    """
    Lazy-loads model when prediction is requested.
    Returns top3 list and full_probs array, and also returns the model load info string.
    """
    model, source = load_model_cached()
    if model is None:
        raise FileNotFoundError(
            f"No model found. Expected TorchScript at {TORCHSCRIPT_PATH} or state_dict at {STATE_DICT_PATH}."
        )

    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_t)
        # handle different model outputs (scripted vs plain)
        if isinstance(logits, tuple) or hasattr(logits, "__len__") and len(logits) == 2:
            # some models return (logits, aux)
            logits = logits[0]
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    top3_idx = probs.argsort()[-3:][::-1]
    top3 = [(CLASSES[i], float(probs[i])) for i in top3_idx]
    return top3, probs, source

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("Food Image Classifier")
st.write("Upload a food image and let the model classify it!")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Provide a manual predict button to avoid auto-blocking when upload occurs
    if st.button("Classify image"):
        with st.spinner("Loading model and running inference (this may take a few seconds)..."):
            try:
                top3, full_probs, source = predict(img)
            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.exception(f"Error during inference: {e}")
            else:
                st.success(f"Model loaded from: {source}")
                st.subheader("Top Prediction")
                st.write(f"**{top3[0][0]}** — {top3[0][1]*100:.2f}% confidence")

                st.subheader("Other likely predictions")
                for cls, prob in top3[1:]:
                    st.write(f"- **{cls}** — {prob*100:.2f}%")

                with st.expander("See full probability distribution"):
                    prob_dict = {CLASSES[i]: float(full_probs[i]) for i in range(len(CLASSES))}
                    st.json(prob_dict)

st.write("---")
st.caption("Built using PyTorch, Timm & Streamlit")