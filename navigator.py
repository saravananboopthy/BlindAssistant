"""
Cross-platform navigation module using Google Maps API.
Works on any OS (no Windows-specific dependencies).
"""

import os
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def get_maps_client():
    try:
        import googlemaps
        api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not api_key:
            return None
        return googlemaps.Client(key=api_key)
    except Exception:
        return None


def clean_html(text):
    return re.sub(r"<.*?>", "", text).replace("&nbsp;", " ").strip()


def get_walking_directions(source, destination):
    """Get walking directions. Returns (result_dict, error_string)."""
    gmaps = get_maps_client()
    if not gmaps:
        return None, "Google Maps API key not configured. Add GOOGLE_MAPS_API_KEY to secrets."

    try:
        routes = gmaps.directions(
            source,
            destination,
            mode="walking",
            departure_time=datetime.now()
        )

        if not routes:
            return None, "No route found between these locations."

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
            "start_address": leg.get("start_address", source),
            "end_address": leg.get("end_address", destination),
        }

        return {"steps": steps, "summary": summary}, None

    except Exception as e:
        return None, f"Navigation error: {str(e)}"
