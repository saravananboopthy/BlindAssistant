"""
Google Maps walking navigation helper
"""

import streamlit as st
import googlemaps
import re
from datetime import datetime


# ---------------- MAP CLIENT ----------------

def get_maps_client():

    api_key = st.secrets.get("GOOGLE_MAPS_API_KEY")

    if not api_key:
        return None

    try:
        return googlemaps.Client(key=api_key)
    except Exception:
        return None


# ---------------- CLEAN HTML ----------------

def clean_html(text):

    text = re.sub(r"<.*?>", "", text)
    text = text.replace("&nbsp;", " ")

    return text.strip()


# ---------------- DIRECTIONS ----------------

def get_walking_directions(source, destination):

    gmaps = get_maps_client()

    if not gmaps:
        return None, "Google Maps API key missing."

    try:

        routes = gmaps.directions(
            source,
            destination,
            mode="walking",
            departure_time=datetime.now()
        )

        if not routes:
            return None, "No walking route found."

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
            "start_address": leg.get("start_address"),
            "end_address": leg.get("end_address")
        }

        return {"steps": steps, "summary": summary}, None

    except Exception as e:

        return None, f"Navigation error: {str(e)}"
