"""
Blind Assistant
AI Vision + Voice Navigation
Streamlit Cloud Compatible
"""

import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp"

import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from streamlit_geolocation import streamlit_geolocation
from streamlit_autorefresh import st_autorefresh
import av
import cv2
import threading
from collections import Counter
from ultralytics import YOLO

from navigator import get_walking_directions


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Blind Assistant",
    page_icon="👁️",
    layout="wide"
)


# ---------------- AUTO REFRESH ----------------
# refresh every 3 seconds to update GPS and UI
st_autorefresh(interval=3000, key="refresh")


# ---------------- SESSION STATE ----------------

defaults = {
    "lat": None,
    "lon": None,
    "nav_steps": [],
    "nav_index": 0,
    "nav_active": False,
    "last_spoken": ""
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------- LOAD YOLO ----------------

@st.cache_resource
def load_model():
    return YOLO("yolov8n")

model = load_model()


# ---------------- RTC CONFIG ----------------

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# ---------------- VIDEO PROCESSOR ----------------

class BlindProcessor(VideoProcessorBase):

    confidence = 0.40

    def __init__(self):
        self.lock = threading.Lock()
        self.detections = {}

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        results = model(img, conf=self.confidence, verbose=False)[0]

        detected = []

        for box in results.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]

            detected.append(label)

            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

            cv2.putText(
                img,
                label,
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

        with self.lock:
            self.detections = dict(Counter(detected))

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------- SPEECH ----------------

def browser_speak(text):

    if text == st.session_state.last_spoken:
        return

    st.session_state.last_spoken = text

    components.html(
        f"""
<script>
const msg = new SpeechSynthesisUtterance("{text}");
msg.rate = 1.2;
msg.pitch = 1;
msg.volume = 1;
speechSynthesis.cancel();
speechSynthesis.speak(msg);
</script>
""",
        height=0
    )


# ---------------- GPS LOCATION ----------------

location = streamlit_geolocation()

if location:

    lat = location.get("latitude")
    lon = location.get("longitude")

    if lat and lon:

        st.session_state.lat = float(lat)
        st.session_state.lon = float(lon)


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("Blind Assistant")

    st.divider()

    st.subheader("📍 Live Location")

    if st.session_state.lat:

        st.success(
            f"{st.session_state.lat:.5f}, {st.session_state.lon:.5f}"
        )

    else:

        st.warning("Allow location permission in browser")


    st.divider()

    st.subheader("Navigation")

    destination = st.text_input("Destination")


    # voice destination
    components.html(
"""
<button onclick="startSpeech()">🎤 Speak Destination</button>

<script>

function startSpeech(){

const recognition = new webkitSpeechRecognition();

recognition.lang="en-US";

recognition.onresult=function(event){

const text = event.results[0][0].transcript;

const inputs = window.parent.document.querySelectorAll("input");

if(inputs.length>0){

inputs[0].value=text;
inputs[0].dispatchEvent(new Event("input",{bubbles:true}));

}

};

recognition.start();

}

</script>
""",
height=80
    )


    if st.button("Start Navigation"):

        if st.session_state.lat and destination:

            source = f"{st.session_state.lat},{st.session_state.lon}"

            result, error = get_walking_directions(source, destination)

            if result:

                st.session_state.nav_steps = result["steps"]
                st.session_state.nav_index = 0
                st.session_state.nav_active = True

            else:

                st.error(error)

        else:

            st.warning("Location or destination missing")


# ---------------- MAIN UI ----------------

st.title("👁 Blind Assistant")

col1, col2 = st.columns([3,2])


# CAMERA

with col1:

    ctx = webrtc_streamer(
        key="blind-assistant",
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=BlindProcessor,
        media_stream_constraints={"video":True,"audio":False},
        async_processing=True
    )


# DETECTIONS

with col2:

    st.subheader("Detected Objects")

    if ctx.state.playing and ctx.video_processor:

        with ctx.video_processor.lock:
            detections = ctx.video_processor.detections.copy()

        if detections:

            text = ", ".join(
                f"{count} {obj}" for obj,count in detections.items()
            )

            st.write(text)

            browser_speak(text)

        else:

            st.info("No objects detected")

    else:

        st.info("Click START to activate camera")


# ---------------- NAVIGATION ----------------

if st.session_state.nav_active:

    st.divider()

    steps = st.session_state.nav_steps
    idx = st.session_state.nav_index

    step = steps[idx]

    st.success(step["text"])

    c1, c2 = st.columns(2)

    if c1.button("Previous") and idx>0:

        st.session_state.nav_index -= 1
        st.rerun()

    if c2.button("Next") and idx < len(steps)-1:

        st.session_state.nav_index += 1
        st.rerun()
