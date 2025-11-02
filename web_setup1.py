import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
from PIL import Image
import tempfile
from streamlit_option_menu import option_menu
import cv2
st.write("OpenCV version:", cv2.__version__)

# --------------------------- PAGE CONFIG --------------------------- #
st.set_page_config(page_title="Animals Detection", page_icon="🤖", layout="wide")

# --------------------------- SIDEBAR --------------------------- #
with st.sidebar:
    st.image("https://ultralytics.com/images/logo.svg", width=120)
    st.title("⚙️ Settings")

    confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05)

    st.markdown("---")
    st.caption("Developed by Mayank Sharma using YOLOv11 + Streamlit")

# --------------------------- LOAD MODEL --------------------------- #
model = YOLO("yolo11n_custom.pt")  # Replace with your trained model path

# --------------------------- HEADER --------------------------- #
st.markdown("<h1 style='text-align:center;'>🧠 YOLOv11 Animals Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Upload an image or use your webcam for real-time detection</p>", unsafe_allow_html=True)
st.markdown("---")

# --------------------------- NAVIGATION MENU --------------------------- #
selected = option_menu(
    menu_title=None,
    options=["Upload Image", "Use Webcam"],
    icons=["image", "camera-video"],
    orientation="horizontal",
)

# --------------------------- IMAGE UPLOAD MODE --------------------------- #
if selected == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        img = cv2.imread(tfile.name)
        results = model(img, conf=confidence)
        annotated_frame = results[0].plot()

        st.image(annotated_frame, channels="BGR", use_column_width=True)

        classes = results[0].boxes.cls
        if len(classes) > 0:
            detected = [model.names[int(c)] for c in classes]
            st.success(f"✅ Detected {len(classes)} object(s):")
            st.write(list(set(detected)))
        else:
            st.warning("No objects detected.")

# --------------------------- WEBCAM MODE (WORKS ONLINE) --------------------------- #
elif selected == "Use Webcam":
    st.markdown("### 🎥 Live Animal Detection")

    class VideoProcessor(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img, conf=confidence)
            annotated = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="animal-detect",
        video_transformer_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

# --------------------------- STYLE SECTION --------------------------- #
st.markdown("""
<style>
[data-testid="stAppViewContainer"] > .main {
    background: linear-gradient(135deg, #f0f2f6 0%, #e0eafc 100%);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e3f2fd 0%, #ffffff 100%);
}
.stButton button {
    background-color: #008CBA;
    color: white;
    border-radius: 10px;
    font-size: 16px;
    font-weight: bold;
    padding: 0.5em 1em;
    transition: 0.3s;
}
.stButton button:hover {
    background-color: #006f9a;
    transform: scale(1.05);
}
h1 {
    color: #2b2b2b;
    font-weight: 700;
}
.stSuccess, .stWarning, .stInfo {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)








