"""A compact Streamlit desktop-style client for the NoteVision FastAPI service."""
from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000").rstrip("/")
SUPPORTED = ("whole_note", "half_note", "quarter_note", "eighth_note", "sixteenth_note")


def api_health() -> tuple[bool, str | None]:
    """Return a user-safe connection state; never expose transport details."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        payload = response.json()
        return response.status_code == 200 and payload.get("status") == "ok", payload.get("model_version")
    except (requests.RequestException, ValueError):
        return False, None


def probability_rows(probabilities: dict[str, float], predicted: str) -> str:
    labels = {"whole_note": "Whole", "half_note": "Half", "quarter_note": "Quarter",
              "eighth_note": "Eighth", "sixteenth_note": "Sixteenth"}
    rows: list[str] = []
    for label in SUPPORTED:
        value = float(probabilities.get(label, 0.0))
        selected = " selected" if label == predicted else ""
        rows.append(
            f'<div class="probability-row{selected}"><span>{labels[label]}</span>'
            f'<div class="probability-track"><div class="probability-fill" style="width:{value * 100:.2f}%"></div></div>'
            f'<strong>{value:.1%}</strong></div>'
        )
    return "".join(rows)


CSS = """
<style>
#MainMenu, header, footer, [data-testid="stDecoration"] { visibility: hidden; display: none; }
.stApp { background: #ECEEF1; color: #20242C; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; }
.block-container { max-width: 1120px; padding: 0 28px 20px; }
.topbar { background: #252A34; margin: 0 -28px 26px; padding: 12px 28px; min-height: 62px; color: #FFFFFF; display: flex; align-items: center; justify-content: space-between; }
.brand { display: flex; align-items: center; }
.brand-name { font-size: 18px; font-weight: 650; line-height: 1.1; }.brand-subtitle { font-size: 12px; color: #C9CDD4; margin-top: 2px; }
.api-status { font-size: 12px; color: #DEE1E6; display: flex; gap: 7px; align-items: center; }.status-dot { width: 8px; height: 8px; border-radius: 50%; background: #8BA36A; display: inline-block; }.status-dot.offline { background: #A96C6C; }
.panel-title { font-size: 15px; font-weight: 650; margin: 1px 0 4px; color: #20242C; }.panel-help, .muted { font-size: 13px; color: #69707D; margin-bottom: 15px; }
[data-testid="stVerticalBlockBorderWrapper"] { background: #FFFFFF; border: 1px solid #D4D8DE; border-radius: 5px; box-shadow: none; }[data-testid="stVerticalBlockBorderWrapper"] > div { padding: 18px; }
[data-testid="stFileUploaderDropzone"] { background: #FFFFFF; border: 1px dashed #B9BEC7; border-radius: 4px; }
.preview, .staff-empty { border: 1px solid #D4D8DE; border-radius: 4px; background-color: #FFFFFF; min-height: 238px; display: flex; align-items: center; justify-content: center; margin: 12px 0 15px; }.staff-empty { color: #69707D; font-size: 13px; position: relative; overflow: hidden; }.staff-lines { width: 88%; height: 54px; position: absolute; display: flex; flex-direction: column; justify-content: space-between; }.staff-lines i { display: block; border-top: 1px solid #E1E4E8; }.staff-caption { position: relative; background: #FFFFFF; padding: 0 9px; }.preview img { max-height: 226px; width: auto; object-fit: contain; }
.stButton > button { background: #7226E8; color: #FFFFFF; border: 1px solid #6220CB; border-radius: 4px; padding: 0.42rem 1rem; font-weight: 600; box-shadow: none; }.stButton > button:hover { background: #6220CB; border-color: #6220CB; color: #FFFFFF; }.stButton > button:focus-visible { outline: 3px solid #B99AF2; outline-offset: 2px; }
.result-name { font-size: 26px; line-height: 1.15; font-weight: 650; margin: 22px 0 4px; color: #20242C; }.result-ru { color: #69707D; font-size: 15px; margin-bottom: 18px; }.confidence { border-top: 1px solid #E2E5E9; border-bottom: 1px solid #E2E5E9; padding: 12px 0; margin-bottom: 16px; font-size: 13px; color: #69707D; }.confidence strong { float: right; color: #20242C; font-size: 17px; }
.probability-row { display: grid; grid-template-columns: 76px 1fr 44px; gap: 8px; align-items: center; font-size: 12px; margin: 10px 0; color: #69707D; }.probability-row strong { color: #505762; text-align: right; font-weight: 600; }.probability-track { height: 7px; background: #E2E5E9; border-radius: 2px; overflow: hidden; }.probability-fill { height: 100%; background: #89909A; }.probability-row.selected { color: #20242C; font-weight: 650; }.probability-row.selected .probability-fill { background: #7226E8; }.model-version { margin-top: 18px; color: #69707D; font-size: 11px; }.supported { border-top: 1px solid #D4D8DE; color: #69707D; margin-top: 25px; padding-top: 12px; font-size: 12px; }
@media (max-width: 768px) { .block-container { padding: 0 14px 16px; }.topbar { margin: 0 -14px 18px; padding: 12px 14px; }.probability-row { grid-template-columns: 68px 1fr 40px; } }
</style>
"""

st.set_page_config(page_title="NoteVision", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
connected, health_version = api_health()
dot_class, status_text = ("", "API connected") if connected else (" offline", "API unavailable")
st.markdown(
    f'<div class="topbar"><div class="brand"><div><div class="brand-name">NoteVision</div>'
    '<div class="brand-subtitle">Note duration recognition</div></div></div>'
    f'<div class="api-status"><span class="status-dot{dot_class}"></span>{status_text}</div></div>', unsafe_allow_html=True,
)

if "recognition_result" not in st.session_state:
    st.session_state.recognition_result = None

input_column, result_column = st.columns((1.15, 0.85), gap="large")
with input_column:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Input</div><div class="panel-help">Select a cropped musical note image in PNG or JPEG format.</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Note image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        if uploaded is None:
            st.markdown('<div class="staff-empty"><div class="staff-lines"><i></i><i></i><i></i><i></i><i></i></div><span class="staff-caption">No image selected</span></div>', unsafe_allow_html=True)
        else:
            image_bytes = uploaded.getvalue()
            st.markdown('<div class="preview">', unsafe_allow_html=True)
            st.image(image_bytes, caption="Selected note", use_container_width=False)
            st.markdown('</div>', unsafe_allow_html=True)
            if st.button("Recognize", type="primary"):
                if Path(uploaded.name).suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    st.error("Select a PNG or JPEG image.")
                elif not connected:
                    st.error("The API is currently unavailable.")
                else:
                    try:
                        response = requests.post(f"{API_URL}/predict", files={"file": (uploaded.name, image_bytes, uploaded.type)}, timeout=20)
                        if response.status_code == 200:
                            st.session_state.recognition_result = response.json()
                        else:
                            st.error("The image could not be recognized.")
                    except requests.RequestException:
                        st.error("The API is currently unavailable.")

with result_column:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Recognition</div>', unsafe_allow_html=True)
        result = st.session_state.recognition_result
        if not result:
            st.markdown('<div class="muted" style="margin-top:24px">No result</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-name">{result["display_name"]}</div><div class="result-ru">{result["display_name_ru"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence">Confidence <strong>{result["confidence"]:.1%}</strong></div>', unsafe_allow_html=True)
            st.markdown(probability_rows(result.get("probabilities", {}), result["class"]), unsafe_allow_html=True)
            st.markdown(f'<div class="model-version">Model version: {result.get("model_version", health_version or "unknown")}</div>', unsafe_allow_html=True)

st.markdown('<div class="supported">Supported values: Whole · Half · Quarter · Eighth · Sixteenth</div>', unsafe_allow_html=True)
