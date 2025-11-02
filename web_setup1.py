import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
from streamlit_option_menu import option_menu

# --------------------------- PAGE CONFIG --------------------------- #
st.set_page_config(page_title="Animals Detection", page_icon="🤖", layout="wide")

# --------------------------- SIDEBAR --------------------------- #
with st.sidebar:
    st.image("https://ultralytics.com/images/logo.svg", width=120)
    st.title("⚙️ Settings")

    confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05)

    st.markdown("---")
    st.subheader("🎥 Camera Settings")

    # ✅ Camera switch option
    cam_index = st.selectbox(
        "Select Camera",
        options=[0, 1, 2, 3],
        format_func=lambda x: f"Camera {x}",
        help="Try switching if your external webcam isn’t showing up."
    )

    st.markdown("---")
    st.caption("Developed by Mayank Sharma using YOLOv11 + Streamlit")

# --------------------------- LOAD MODEL --------------------------- #
model = YOLO("D:/yolo11/yolo11n_custom.pt")  # 🔁 Replace with your trained model path

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

        # Display detected classes
        classes = results[0].boxes.cls
        if len(classes) > 0:
            detected = [model.names[int(c)] for c in classes]
            st.success(f"✅ Detected {len(classes)} object(s):")
            st.write(list(set(detected)))
        else:
            st.warning("No objects detected.")

# --------------------------- WEBCAM MODE --------------------------- #
elif selected == "Use Webcam":
    if "run_webcam" not in st.session_state:
        st.session_state.run_webcam = False

    start_btn = st.button("▶ Start Webcam")
    stop_btn = st.button("⛔ Stop Webcam")

    if start_btn:
        st.session_state.run_webcam = True
    if stop_btn:
        st.session_state.run_webcam = False

    stframe = st.empty()

    if st.session_state.run_webcam:
        cap = cv2.VideoCapture(cam_index)

        if not cap.isOpened():
            st.error(f"⚠️ Could not open Camera {cam_index}. Try another one.")
        else:
            st.info(f"📷 Using Camera {cam_index}")
            while st.session_state.run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Unable to access webcam feed.")
                    break

                results = model(frame, conf=confidence)
                annotated_frame = results[0].plot()
                stframe.image(annotated_frame, channels="BGR", use_column_width=True)

            cap.release()
            st.success("✅ Webcam stopped successfully.")

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


