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
import math

from navigator import get_walking_directions


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Blind Assistant",
    page_icon="👁️",
    layout="wide"
)


# ---------------- AUTO REFRESH ----------------

st_autorefresh(interval=1500, key="refresh")


# ---------------- SESSION STATE ----------------

defaults = {
    "lat": None,
    "lon": None,
    "nav_steps": [],
    "nav_index": 0,
    "nav_active": False,
    "last_spoken": "",
    "last_nav_spoken": "",
    "destination_input": ""
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------- LOAD YOLO ----------------

@st.cache_resource
def load_model():
    return YOLO("yolov8s")

model = load_model()


# ---------------- RTC CONFIG ----------------

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# ---------------- VIDEO PROCESSOR ----------------

class BlindProcessor(VideoProcessorBase):

    confidence = 0.60

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

    components.html(
        f"""
<script>
const msg = new SpeechSynthesisUtterance("{text}");
msg.rate = 1.1;
msg.pitch = 1;
msg.volume = 1;
window.speechSynthesis.speak(msg);
</script>
""",
        height=0
    )


# ---------------- GPS LOCATION ----------------

location = streamlit_geolocation()

if location:

    lat = location.get("latitude")
    lon = location.get("longitude")

    if lat is not None and lon is not None:

        st.session_state.lat = float(lat)
        st.session_state.lon = float(lon)


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("Blind Assistant")

    st.divider()

    st.subheader("📍 Live Location")

    if st.session_state.lat is not None:

        st.success(
            f"{st.session_state.lat:.5f}, {st.session_state.lon:.5f}"
        )

    else:

        st.warning("Allow location permission in browser")

    st.divider()

    st.subheader("Navigation")

    destination = st.text_input(
        "Destination",
        key="destination_input"
    )

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
inputs[0].value = text;
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

        destination = st.session_state.destination_input

        if st.session_state.lat is not None and destination.strip() != "":

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

        important = {"person","car","bus","truck","bicycle","motorcycle","dog"}

        filtered = []

        for obj,count in detections.items():
            if obj in important:
                filtered.append(f"{count} {obj}")

        text = ", ".join(filtered)

        st.write(text)

        if text and text != st.session_state.last_spoken:
            browser_speak(text)
            st.session_state.last_spoken = text

    else:

        st.info("Click START to activate camera")


# ---------------- NAVIGATION ----------------

def distance_meters(lat1, lon1, lat2, lon2):

    R = 6371000

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2

    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))


if st.session_state.nav_active:

    st.divider()

    steps = st.session_state.nav_steps
    idx = st.session_state.nav_index

    if idx < len(steps):

        step = steps[idx]

        st.success(step["text"])

        if st.session_state.lat is not None:

            distance = distance_meters(
                st.session_state.lat,
                st.session_state.lon,
                step["lat"],
                step["lon"]
            )

            st.info(f"Distance to next step: {int(distance)} meters")

            if distance <= 25:

                nav_text = "Navigation " + step["text"]

                if nav_text != st.session_state.last_nav_spoken:

                    browser_speak(nav_text)

                    st.session_state.last_nav_spoken = nav_text

                    st.session_state.nav_index += 1

    else:

        st.success("Destination reached")

        if "Destination reached" != st.session_state.last_nav_spoken:
            browser_speak("Destination reached")
            st.session_state.last_nav_spoken = "Destination reached"
