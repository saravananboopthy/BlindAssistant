"""
Blind Assistant
AI Vision + Voice Navigation
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
import time
import math
from collections import Counter
from ultralytics import YOLO

from navigator import get_walking_directions


# ---------------- PAGE ----------------

st.set_page_config(
    page_title="Blind Assistant",
    page_icon="👁️",
    layout="wide"
)

# force UI refresh every 0.4 sec
st_autorefresh(interval=400, key="refresh")


# ---------------- SESSION ----------------

defaults = {
    "lat": None,
    "lon": None,
    "nav_steps": [],
    "nav_index": 0,
    "nav_active": False,
    "destination_input": "",
    "last_navigation": "",
    "speech_queue": []
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------- SPEECH ----------------

def speak(text):
    st.session_state.speech_queue.append(text)


def run_speech():
    if st.session_state.speech_queue:

        msg = st.session_state.speech_queue.pop(0)

        components.html(
            f"""
<script>
const msg = new SpeechSynthesisUtterance("{msg}");
msg.rate = 1.1;
msg.pitch = 1;
speechSynthesis.speak(msg);
</script>
""",
            height=0
        )


# ---------------- YOLO ----------------

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()


# ---------------- RTC ----------------

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}
)


# ---------------- VIDEO PROCESSOR ----------------

class BlindProcessor(VideoProcessorBase):

    def __init__(self):

        self.lock = threading.Lock()
        self.detections = {}

        self.last_spoken = ""
        self.last_time = 0


    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        results = model(img, conf=0.45, verbose=False)[0]

        detected = []

        for box in results.boxes:

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]

            detected.append(label)

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.putText(
                img,
                label,
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

        counts = dict(Counter(detected))

        with self.lock:
            self.detections = counts

        if counts:

            text = ", ".join(counts.keys())

            now = time.time()

            if text != self.last_spoken and now - self.last_time > 1:

                speak(text)

                self.last_spoken = text
                self.last_time = now

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------- LOCATION ----------------

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

    st.subheader("📍 Live Location")

    if st.session_state.lat:
        st.success(f"{st.session_state.lat:.5f}, {st.session_state.lon:.5f}")
    else:
        st.warning("Allow location permission")

    st.subheader("Navigation")

    destination = st.text_input("Destination", key="destination_input")

    if st.button("Start Navigation"):

        if st.session_state.lat and destination:

            source = f"{st.session_state.lat},{st.session_state.lon}"

            result,error = get_walking_directions(source,destination)

            if result:

                st.session_state.nav_steps = result["steps"]
                st.session_state.nav_index = 0
                st.session_state.nav_active = True

                speak("Navigation started")

            else:
                st.error(error)


# ---------------- MAIN ----------------

st.title("👁 Blind Assistant")

col1,col2 = st.columns([3,2])


# ---------------- CAMERA ----------------

with col1:

    ctx = webrtc_streamer(
        key="camera",
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=BlindProcessor,
        media_stream_constraints={"video":True,"audio":False},
        async_processing=True
    )


# ---------------- DETECTION PANEL ----------------

with col2:

    st.subheader("Detected Objects")

    if ctx.state.playing and ctx.video_processor:

        with ctx.video_processor.lock:
            detections = ctx.video_processor.detections.copy()

        if detections:

            text = ", ".join(
                f"{v} {k}" for k,v in detections.items()
            )

            st.success(text)

        else:
            st.info("No obstacle detected")


# ---------------- NAVIGATION ----------------

def distance_meters(lat1, lon1, lat2, lon2):

    R = 6371000

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2-lat1)
    dlambda = math.radians(lon2-lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2

    return 2 * R * math.atan2(math.sqrt(a),math.sqrt(1-a))


if st.session_state.nav_active:

    steps = st.session_state.nav_steps
    idx = st.session_state.nav_index

    if idx < len(steps):

        step = steps[idx]

        st.success(step["text"])

        if st.session_state.lat:

            distance = distance_meters(
                st.session_state.lat,
                st.session_state.lon,
                step["lat"],
                step["lon"]
            )

            if distance <= 20:

                if step["text"] != st.session_state.last_navigation:

                    speak(step["text"])

                    st.session_state.last_navigation = step["text"]
                    st.session_state.nav_index += 1

    else:

        speak("Destination reached")
        st.success("Destination reached")


# ---------------- SPEECH RUNNER ----------------

run_speech()
