"""Streamlit UI; predictions are intentionally delegated to FastAPI."""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000").rstrip("/")

st.set_page_config(page_title="NoteVision", page_icon="🎵")
st.title("🎵 NoteVision")
st.write("Определяем длительность музыкальной ноты по её изображению через ML API.")
st.caption("Поддерживаются: Whole, Half, Quarter, Eighth и Sixteenth Note.")

uploaded = st.file_uploader("Upload a PNG, JPG, or JPEG image", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    image_bytes = uploaded.getvalue()
    st.image(image_bytes, caption="Uploaded image", use_container_width=True)
    if st.button("Recognize note", type="primary"):
        with st.spinner("Sending image to recognition API..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict", files={"file": (uploaded.name, image_bytes, uploaded.type)}, timeout=20,
                )
                if response.status_code != 200:
                    detail = response.json().get("detail", response.text)
                    st.error(f"API rejected the file: {detail}")
                else:
                    result = response.json()
                    st.success(f"{result['display_name']} — {result['display_name_ru']}")
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                    st.caption(f"Model version: {result['model_version']}")
                    probabilities = pd.DataFrame(
                        result["probabilities"].items(), columns=["Class", "Probability"]
                    ).set_index("Class")
                    st.bar_chart(probabilities)
            except requests.RequestException:
                st.error("Recognition API is unavailable. Check that the api container is healthy and API_URL is correct.")
else:
    st.info("Upload a note image, then select Recognize note.")
