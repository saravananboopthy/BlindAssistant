"""
Google Maps walking navigation helper
"""

import streamlit as st
import googlemaps
import re
from datetime import datetime


def get_maps_client():
    try:
        api_key = st.secrets["GOOGLE_MAPS_API_KEY"]
        return googlemaps.Client(key=api_key)
    except Exception:
        return None


def clean_html(text):
    return re.sub(r"<.*?>", "", text).replace("&nbsp;", " ").strip()


def get_walking_directions(source, destination):

    gmaps = get_maps_client()

    if not gmaps:
        return None, "Google Maps API key missing in Streamlit secrets."

    try:

        routes = gmaps.directions(
            source,
            destination,
            mode="walking",
            departure_time=datetime.now()
        )

        if not routes:
            return None, "No route found."

        leg = routes[0]["legs"][0]

        steps = []

        for s in leg["steps"]:

            dist = int(s["distance"]["value"])

            if dist < 5:
                continue

            instr = clean_html(s["html_instructions"])

            steps.append({
                "instruction": instr,
                "distance": dist,
                "text": f"{instr} for {dist} meters"
            })

        summary = {
            "distance": leg["distance"]["text"],
            "duration": leg["duration"]["text"],
        }

        return {"steps": steps, "summary": summary}, None

    except Exception as e:
        return None, f"Navigation error: {str(e)}"
