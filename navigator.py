"""
Google Maps walking navigation helper
Reliable version
"""

import streamlit as st
import googlemaps
import re


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


# ---------------- FORMAT SOURCE ----------------

def format_source(source):
    """
    Ensures coordinates are valid string format
    """
    if isinstance(source, (list, tuple)):
        return f"{source[0]},{source[1]}"

    return str(source)


# ---------------- DIRECTIONS ----------------

def get_walking_directions(source, destination):

    gmaps = get_maps_client()

    if not gmaps:
        return None, "Google Maps API key missing."

    try:

        # Ensure proper format
        source = format_source(source)
        destination = str(destination)

        routes = gmaps.directions(
            source,
            destination,
            mode="walking"
        )

        if not routes:
            return None, "No walking route found."

        leg = routes[0]["legs"][0]

        steps = []

        for s in leg["steps"]:

            dist = int(s["distance"]["value"])

            # skip tiny instructions
            if dist < 5:
                continue

            instr = clean_html(s["html_instructions"])

      steps.append({
    "instruction": instr,
    "distance": dist,
    "text": f"{instr} for {dist} meters",
    "lat": s["end_location"]["lat"],
    "lon": s["end_location"]["lng"]
})

        summary = {
            "distance": leg["distance"]["text"],
            "duration": leg["duration"]["text"],
            "start_address": leg.get("start_address"),
            "end_address": leg.get("end_address")
        }

        return {"steps": steps, "summary": summary}, None

    except googlemaps.exceptions.ApiError as e:

        return None, f"Google Maps API error: {str(e)}"

    except Exception as e:

        return None, f"Navigation error: {str(e)}"
