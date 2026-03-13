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


# ---------------- PAGE ----------------

st.set_page_config(
    page_title="Blind Assistant",
    page_icon="👁️",
    layout="wide"
)

# refresh slower so speech can play
st_autorefresh(interval=3000, key="refresh")


# ---------------- SESSION ----------------

defaults = {
    "lat": None,
    "lon": None,
    "nav_steps": [],
    "nav_index": 0,
    "nav_active": False,
    "last_detection": "",
    "last_navigation": "",
    "destination_input": ""
}

for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------- LOAD YOLO ----------------

@st.cache_resource
def load_model():
    return YOLO("yolov8n")

model = load_model()


# ---------------- RTC ----------------

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# ---------------- SPEECH ----------------

def speak(text):

    components.html(
    f"""
<script>

if(!window.voiceBusy){{

    window.voiceBusy = true

    const msg = new SpeechSynthesisUtterance("{text}")
    msg.rate = 1.1
    msg.pitch = 1
    msg.volume = 1

    msg.onend = function(){{
        window.voiceBusy = false
    }}

    speechSynthesis.speak(msg)

}}

</script>
""",
    height=0
    )


# ---------------- VIDEO PROCESSOR ----------------

class BlindProcessor(VideoProcessorBase):

    confidence = 0.45

    def __init__(self):
        self.lock = threading.Lock()
        self.detections = {}

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

        with self.lock:
            self.detections = dict(Counter(detected))

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

    st.divider()

    st.subheader("📍 Live Location")

    if st.session_state.lat:

        st.success(
            f"{st.session_state.lat:.5f}, {st.session_state.lon:.5f}"
        )

    else:

        st.warning("Allow GPS in browser")


    st.divider()

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

            else:

                st.error(error)

        else:

            st.warning("Location or destination missing")


# ---------------- MAIN ----------------

st.title("👁 Blind Assistant")

col1,col2 = st.columns([3,2])


# CAMERA

with col1:

    ctx = webrtc_streamer(
        key="camera",
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=BlindProcessor,
        media_stream_constraints={"video":True,"audio":False},
        async_processing=True
    )


# ---------------- DETECTIONS ----------------

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

            if text != st.session_state.last_detection:

                speak(text)

                st.session_state.last_detection = text

        else:

            st.write("Nothing detected")


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

        if st.session_state.last_navigation != "done":

            speak("Destination reached")

            st.session_state.last_navigation = "done"
