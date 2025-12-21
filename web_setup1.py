import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Animals Detection", page_icon="🤖", layout="wide")

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.image("https://ultralytics.com/images/logo.svg", width=120)
    st.title("⚙️ Settings")
    confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05)
    st.caption("Developed by Mayank Sharma using YOLOv11 + Streamlit")

# ---------------- LOAD MODEL ---------------- #
model = YOLO("yolo11n_custom.pt")  # MUST be inside repo

# ---------------- HEADER ---------------- #
st.markdown("<h1 style='text-align:center;'>🧠 YOLOv11 Animals Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload an image to detect animals</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- MENU ---------------- #
selected = option_menu(
    None,
    ["Upload Image", "Use Webcam"],
    icons=["image", "camera-video"],
    orientation="horizontal",
)

# ---------------- IMAGE UPLOAD ---------------- #
if selected == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        img = cv2.imread(tfile.name)
        results = model(img, conf=confidence)
        annotated = results[0].plot()

        st.image(annotated, channels="BGR", use_column_width=True)

        classes = results[0].boxes.cls
        if len(classes) > 0:
            names = [model.names[int(c)] for c in classes]
            st.success(f"Detected: {set(names)}")
        else:
            st.warning("No animals detected.")

# ---------------- WEBCAM (DISABLED) ---------------- #
elif selected == "Use Webcam":
    st.warning("🚫 Webcam is not supported on Streamlit Cloud.")
    st.info("Please use Image Upload instead.")
