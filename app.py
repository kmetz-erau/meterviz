import math
import os
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image

DATA_FILE = "meter_readings.csv"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# If needed on your host, set the Tesseract path here:
# pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

st.set_page_config(page_title="Meter Reader", layout="wide")
st.title("Meter Reading App")
st.write("Capture or upload a meter photo, enter details, and read digital or analog meters.")

def preprocess_ocr(image_np: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return thresh

def read_digital_meter(image_np: np.ndarray) -> str:
    processed = preprocess_ocr(image_np)
    config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(processed, config=config)
    cleaned = "".join(ch for ch in text if ch.isdigit() or ch in [".", "-"])
    return cleaned.strip()

def detect_circle(gray: np.ndarray):
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=40,
        maxRadius=0,
    )
    if circles is None:
        return None
    circles = np.round(circles[0, :]).astype("int")
    # choose the biggest circle found
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    return circles[0]  # x, y, r

def angle_from_center(cx, cy, x, y):
    ang = math.degrees(math.atan2(cy - y, x - cx))
    if ang < 0:
        ang += 360
    return ang

def map_angle_to_value(angle, min_angle, max_angle, min_value, max_value):
    """
    Maps an angle on the dial to a meter value.
    Handles wrap-around if the scale crosses 0 degrees.
    """
    if max_angle <= min_angle:
        max_angle += 360
    if angle < min_angle:
        angle += 360

    angle = max(min(angle, max_angle), min_angle)
    ratio = (angle - min_angle) / (max_angle - min_angle)
    return min_value + ratio * (max_value - min_value)

def read_analog_meter(
    image_np: np.ndarray,
    min_value: float,
    max_value: float,
    min_angle: float,
    max_angle: float,
):
    """
    Simple analog meter reader:
    1) detect dial circle
    2) detect line segments
    3) pick the segment most likely to be the needle
    4) map needle angle to a value

    This is a starter approach and works best when the camera is fairly square
    to the dial and the needle is clearly visible.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    circle = detect_circle(gray)
    if circle is None:
        return "", None

    cx, cy, r = circle

    edges = cv2.Canny(gray, 50, 150)
    mask = np.zeros_like(edges)
    cv2.circle(mask, (cx, cy), int(r * 0.95), 255, thickness=-1)
    edges = cv2.bitwise_and(edges, edges, mask=mask)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=int(r * 0.35),
        maxLineGap=15,
    )

    if lines is None:
        return "", (cx, cy, r)

    best_line = None
    best_score = -1

    for line in lines:
        x1, y1, x2, y2 = line[0]

        d1 = math.hypot(x1 - cx, y1 - cy)
        d2 = math.hypot(x2 - cx, y2 - cy)
        dist_to_center = min(d1, d2)
        length = math.hypot(x2 - x1, y2 - y1)

        # favor long lines that pass near the center
        score = length - dist_to_center

        if dist_to_center < r * 0.35 and score > best_score:
            best_score = score
            best_line = (x1, y1, x2, y2)

    if best_line is None:
        return "", (cx, cy, r)

    x1, y1, x2, y2 = best_line
    d1 = math.hypot(x1 - cx, y1 - cy)
    d2 = math.hypot(x2 - cx, y2 - cy)
    needle_tip = (x1, y1) if d1 > d2 else (x2, y2)

    angle = angle_from_center(cx, cy, needle_tip[0], needle_tip[1])
    reading = map_angle_to_value(angle, min_angle, max_angle, min_value, max_value)

    return f"{reading:.2f}", (cx, cy, r, best_line, angle)

def save_reading(record: dict):
    df_new = pd.DataFrame([record])

    if os.path.exists(DATA_FILE):
        df_old = pd.read_csv(DATA_FILE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(DATA_FILE, index=False)

col1, col2 = st.columns(2)

with col1:
    meter_name = st.text_input("Meter name", placeholder="Example: Main water meter")

with col2:
    location = st.text_input("Location", placeholder="Example: Basement utility room")

meter_type = st.radio("Meter type", ["Auto", "Digital", "Analog"], horizontal=True)

timestamp = st.text_input(
    "Timestamp",
    value=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
)

mode_help = st.empty()

analog_min_value = 0.0
analog_max_value = 100.0
analog_min_angle = 225.0
analog_max_angle = 315.0

if meter_type == "Analog":
    st.subheader("Analog calibration")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        analog_min_value = st.number_input("Dial min value", value=0.0)
    with c2:
        analog_max_value = st.number_input("Dial max value", value=100.0)
    with c3:
        analog_min_angle = st.number_input("Needle angle at min value", value=225.0)
    with c4:
        analog_max_angle = st.number_input("Needle angle at max value", value=315.0)

uploaded_file = st.camera_input("Take a photo of the meter") or st.file_uploader(
    "Or upload an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image_np, caption="Meter image", use_container_width=True)

    if st.button("Read and Save Meter"):
        if not meter_name.strip():
            st.error("Please enter a meter name.")
        elif not location.strip():
            st.error("Please enter a location.")
        else:
            with st.spinner("Reading meter..."):
                reading = ""
                debug = None
                used_method = ""

                if meter_type == "Digital":
                    reading = read_digital_meter(image_np)
                    used_method = "digital_ocr"

                elif meter_type == "Analog":
                    reading, debug = read_analog_meter(
                        image_np,
                        analog_min_value,
                        analog_max_value,
                        analog_min_angle,
                        analog_max_angle,
                    )
                    used_method = "analog_needle"

                else:
                    reading = read_digital_meter(image_np)
                    used_method = "digital_ocr"
                    if not reading:
                        reading, debug = read_analog_meter(
                            image_np,
                            analog_min_value,
                            analog_max_value,
                            analog_min_angle,
                            analog_max_angle,
                        )
                        used_method = "analog_needle_fallback"

            image_filename = f"meter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            image_path = os.path.join(UPLOAD_DIR, image_filename)
            image.save(image_path)

            record = {
                "meter_name": meter_name,
                "location": location,
                "meter_type": meter_type,
                "reading": reading,
                "method_used": used_method,
                "timestamp": timestamp,
                "image_file": image_path,
            }
            save_reading(record)

            st.success("Saved successfully")
            st.write(f"**Meter name:** {meter_name}")
            st.write(f"**Location:** {location}")
            st.write(f"**Timestamp:** {timestamp}")
            st.write(f"**Reading:** {reading if reading else 'No reading detected'}")
            st.write(f"**Method used:** {used_method}")

            if debug and meter_type == "Analog":
                cx, cy, r, best_line, angle = debug
                x1, y1, x2, y2 = best_line
                debug_img = image_np.copy()
                cv2.circle(debug_img, (cx, cy), r, (255, 0, 0), 2)
                cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.circle(debug_img, (cx, cy), 4, (0, 0, 255), -1)
                st.image(debug_img, caption=f"Analog debug view, angle: {angle:.1f}°", use_container_width=True)

st.divider()
st.subheader("Saved readings")

if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    st.dataframe(df, use_container_width=True)
else:
    st.info("No saved readings yet.")
