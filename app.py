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
import av
import cv2
import threading
from collections import Counter
from ultralytics import YOLO
import math
import time

from navigator import get_walking_directions


# ---------------- PAGE ----------------

st.set_page_config(
    page_title="Blind Assistant",
    page_icon="👁️",
    layout="wide"
)


# ---------------- SESSION ----------------

defaults = {
    "lat": None,
    "lon": None,
    "nav_steps": [],
    "nav_index": 0,
    "nav_active": False,
    "last_navigation": "",
    "destination_input": ""
}

for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------- SPEECH ----------------

def speak(text):

    components.html(
        f"""
<script>
const msg = new SpeechSynthesisUtterance("{text}");
msg.rate = 1.1;
msg.pitch = 1;
msg.volume = 1;
speechSynthesis.speak(msg);
</script>
""",
        height=0
    )


# ---------------- LOAD YOLO ----------------

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()


# ---------------- RTC ----------------

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# ---------------- VIDEO PROCESSOR ----------------

class BlindProcessor(VideoProcessorBase):

    confidence = 0.45

    def __init__(self):
        self.lock = threading.Lock()
        self.detections = {}
        self.last_spoken = ""
        self.last_time = 0
        self.speak_now = None

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        results = model(img, conf=self.confidence, verbose=False)[0]

        detected = []

        for box in results.boxes:

            x1,y1,x2,y2 = map(int,box.xyxy[0])
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

            if text != self.last_spoken and (now - self.last_time) > 1:

                self.speak_now = text
                self.last_spoken = text
                self.last_time = now

        return av.VideoFrame.from_ndarray(img,format="bgr24")


# ---------------- GPS ----------------

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

        st.success(
            f"{st.session_state.lat:.5f}, {st.session_state.lon:.5f}"
        )

    else:

        st.warning("Allow location permission")

    st.subheader("Navigation")

    destination = st.text_input(
        "Destination",
        key="destination_input"
    )

    if st.button("Start Navigation"):

        destination = st.session_state.destination_input

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

        else:

            st.warning("Location or destination missing")


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

    detection_box = st.empty()

    if ctx.state.playing and ctx.video_processor:

        with ctx.video_processor.lock:

            detections = ctx.video_processor.detections.copy()
            speak_now = ctx.video_processor.speak_now
            ctx.video_processor.speak_now = None

        if detections:

            text = ", ".join(
                f"{count} {obj}" for obj,count in detections.items()
            )

            detection_box.success(text)

        else:

            detection_box.info("No obstacle detected")

        if speak_now:

            speak(speak_now)


# ---------------- NAVIGATION ----------------

def distance_meters(lat1, lon1, lat2, lon2):

    R = 6371000

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2

    return 2 * R * math.atan2(math.sqrt(a),math.sqrt(1-a))


if st.session_state.nav_active:

    st.divider()

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

            st.info(f"Distance: {int(distance)} meters")

            if distance <= 20:

                nav_text = step["text"]

                if nav_text != st.session_state.last_navigation:

                    speak(nav_text)

                    st.session_state.last_navigation = nav_text

                    st.session_state.nav_index += 1

    else:

        st.success("Destination reached")

        speak("Destination reached")
