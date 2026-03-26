import streamlit as st
import cv2
import os
import subprocess
import json
import matplotlib.pyplot as plt
import shutil
import pandas as pd
import random

# FULL WIDTH UI
st.set_page_config(layout="wide")

FFPROBE_PATH = r"C:\Users\Rashmitha\Downloads\MAA_Video_format_Classification\MAA_Video_format_Classification\ffmpeg-8.1-essentials_build\bin\ffprobe.exe"

# =========================
# BACKEND FUNCTIONS
# =========================

def clear_output():
    if os.path.exists("output"):
        shutil.rmtree("output")
    os.makedirs("output", exist_ok=True)


def get_duration(video):
    cmd = [
        FFPROBE_PATH,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    return round(float(result.stdout.strip()), 2)


def extract_frames(video):
    cap = cv2.VideoCapture(video)
    frames, count = [], 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        name = f"frame_{count}.jpg"
        cv2.imwrite(f"output/{name}", frame)
        frames.append(name)
        count += 1

    cap.release()
    return frames, count


def get_types(video):
    cmd = [
        FFPROBE_PATH,
        "-select_streams", "v",
        "-show_frames",
        "-print_format", "json",
        video
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    data = json.loads(result.stdout)

    types, sizes = [], []

    for f in data.get("frames", []):
        t = f.get("pict_type")
        s = f.get("pkt_size", "0")

        if t in ["I", "P", "B"]:
            types.append(t)
            sizes.append(int(s) if s.isdigit() else 0)

    return types, sizes


def classify(frames, types):
    os.makedirs("output/I", exist_ok=True)
    os.makedirs("output/P", exist_ok=True)
    os.makedirs("output/B", exist_ok=True)

    i = p = b = 0
    gop = []

    for idx, t in enumerate(types):
        if idx >= len(frames):
            break

        src = f"output/{frames[idx]}"

        if t == "I":
            dst = f"output/I/{frames[idx]}"
            i += 1
        elif t == "P":
            dst = f"output/P/{frames[idx]}"
            p += 1
        else:
            dst = f"output/B/{frames[idx]}"
            b += 1

        os.rename(src, dst)
        gop.append(t)

    return i, p, b, gop


def transmission(gop):
    result, buffer = [], []

    for f in gop:
        if f in ["I", "P"]:
            result.append(f)
            result.extend(buffer)
            buffer = []
        else:
            buffer.append(f)

    result.extend(buffer)
    return result


def split_gops(gop):
    gops = []
    current = []

    for f in gop:
        if f == "I" and current:
            gops.append(current)
            current = []
        current.append(f)

    if current:
        gops.append(current)

    return gops


# =========================
# FRONTEND
# =========================

# =========================
# FRONTEND (UPDATED UI)
# =========================

st.markdown("<h1 style='text-align:center;color:#4CAF50;'>🎥 Video Compression Analyzer</h1>", unsafe_allow_html=True)

file = st.file_uploader("📂 Upload Video", type=["mp4","avi","mkv"])

if file:

    os.makedirs("temp", exist_ok=True)
    path = f"temp/{file.name}"

    with open(path, "wb") as f:
        f.write(file.read())

    st.success("✅ Video Uploaded Successfully")

    clear_output()

    duration = get_duration(path)
    frames, total = extract_frames(path)
    types, sizes = get_types(path)

    if not types:
        st.error("Frame detection failed")
        st.stop()

    i, p, b, gop = classify(frames, types)

    # =========================
    # DASHBOARD METRICS
    # =========================

    st.markdown("### 📊 Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⏱ Duration", f"{duration}s")
    col2.metric("🎞 Total Frames", total)
    col3.metric("🟢 I Frames", i)
    col4.metric("🔵 P Frames", p)

    st.metric("🟡 B Frames", b)

    # =========================
    # DISTRIBUTION
    # =========================

    with st.expander("📊 Frame Distribution (%)", expanded=True):
        total_frames = i + p + b
        st.write(f"I: {round(i/total_frames*100,2)}%")
        st.write(f"P: {round(p/total_frames*100,2)}%")
        st.write(f"B: {round(b/total_frames*100,2)}%")

    # =========================
    # GOP
    # =========================

    with st.expander("📜 GOP Structure", expanded=False):
        st.text_area("Sequence", " → ".join(gop), height=200)

        gops = split_gops(gop)
        st.write(f"Total GOPs: {len(gops)}")

        for idx, g in enumerate(gops):
            st.write(f"GOP {idx+1}:")
            st.text(" → ".join(g))

    # =========================
    # TRANSMISSION
    # =========================

    with st.expander("🔁 Transmission Sequence", expanded=False):
        trans = transmission(gop)
        st.text_area("Transmission Order", " → ".join(trans), height=200)

    # =========================
    # COMPRESSION
    # =========================

    with st.expander("💾 Compression Analysis", expanded=True):

        def avg(t):
            vals = [s for typ, s in zip(types, sizes) if typ == t]
            return sum(vals)//max(1,len(vals))

        i_size = avg("I")
        p_size = avg("P")
        b_size = avg("B")

        ip_size = (i_size + p_size) / 2
        ipb_size = (i_size + p_size + b_size) / 3

        df = pd.DataFrame({
            "Structure": ["I", "IP", "IPB"],
            "Avg Size": [int(i_size), int(ip_size), int(ipb_size)]
        })

        st.dataframe(df, use_container_width=True)

        st.markdown("#### 📉 Compression Ratios")

        I_comp = 1.0
        IP_comp = round(i_size / ip_size, 2)
        IPB_comp = round(i_size / ipb_size, 2)

        st.write(f"I: {I_comp}x | IP: {IP_comp}x | IPB: {IPB_comp}x")

        st.info("Compression improves from I → IP → IPB")

        # 🔥 SIZE COMPARISON
        st.markdown("#### 📊 Size Comparison")

        raw_size = total
        ip_model = i + (p * 0.5)
        ipb_model = i + (p * 0.5) + (b * 0.25)

        df2 = pd.DataFrame({
            "Structure": ["RAW", "I", "IP", "IPB"],
            "Estimated Size": [
                round(raw_size,2),
                round(i,2),
                round(ip_model,2),
                round(ipb_model,2)
            ]
        })

        st.dataframe(df2, use_container_width=True)

    # =========================
    # GRAPH
    # =========================

    st.markdown("### 📈 Frame Distribution Graph")
    fig = plt.figure()
    plt.bar(["I","P","B"], [i,p,b])
    st.pyplot(fig)

    # =========================
    # SAMPLE FRAMES
    # =========================

    st.markdown("### 🖼 Sample Frames")

    c1,c2,c3 = st.columns(3)

    def show(folder,col):
        if os.path.exists(folder):
            f = os.listdir(folder)
            if f:
                img = cv2.imread(os.path.join(folder, random.choice(f)))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                col.image(img, caption=folder)

    show("output/I", c1)
    show("output/P", c2)
    show("output/B", c3)

    st.balloons()