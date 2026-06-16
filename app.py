"""
ColorizeAI Studio — Streamlit Web App (premium AI-product UI)
Author: Aditya Mhetre
Model: Zhang et al. 2016 (OpenCV DNN)
"""

import streamlit as st
import cv2
import numpy as np
import os
import time
import io
import zipfile
import pandas as pd
import altair as alt
from PIL import Image, ImageEnhance

try:
    from streamlit_image_comparison import image_comparison
    HAS_SLIDER = True
except ImportError:
    HAS_SLIDER = False
import gdown

DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(DIR, "model", "colorization_release_v2.caffemodel")

def ensure_model():
    """Download caffemodel from Google Drive on first run (Streamlit Cloud)."""
    if not os.path.exists(MODEL):
        os.makedirs(os.path.join(DIR, "model"), exist_ok=True)
        with st.spinner("Downloading model (first run only, ~140MB)..."):
            gdown.download(
                "https://drive.google.com/file/d/1isy5zFjFx0IYLrc1bFm97RFyXGYtoKm-/view?usp=sharing",
                MODEL,
                quiet=False
            )

ensure_model()
st.set_page_config(page_title="ColorizeAI Studio", layout="wide",
                   initial_sidebar_state="collapsed")

# ─────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root{
    --bg:#000000; --fg:#FFFFFF; --panel:#070707; --panel-2:#0c0c0c;
    --border:#1A1A1A; --border-2:#262626;
    --accent:#CBA6F7; --accent-2:#7c5cff; --accent-hover:#d8bcff;
    --muted:#9a9a9a; --radius:16px;
    --font:'Inter',-apple-system,sans-serif;
}

html, body, .stApp, [class*="css"]{ font-family:var(--font); background:var(--bg); color:var(--fg); }

.main .block-container{ padding:4.5rem 2rem 2rem; max-width:1240px; }
#MainMenu, footer, header{ visibility:hidden; }

/* ── Navbar ── */
.nav{
    position:fixed; top:0; left:0; right:0; height:60px; z-index:1000;
    display:flex; align-items:center; justify-content:space-between;
    padding:0 32px; background:rgba(0,0,0,0.75); backdrop-filter:blur(12px);
    border-bottom:1px solid var(--border);
}
.brand{ font-weight:800; font-size:1.1rem; letter-spacing:-0.02em;
    display:flex; align-items:center; gap:10px; }
.brand .mark{
    width:22px; height:22px; border-radius:7px;
    background:linear-gradient(135deg,var(--accent),var(--accent-2));
    box-shadow:0 0 16px rgba(203,166,247,0.6);
}
.brand .accent{ color:var(--accent); }
.nav-links a{ color:var(--muted); text-decoration:none; margin-left:26px;
    font-size:0.9rem; font-weight:500; transition:color 0.15s; }
.nav-links a:hover{ color:var(--fg); }

/* ── Hero ── */
.hero{
    position:relative; border:1px solid var(--border-2); border-radius:24px;
    padding:3.5rem 3rem; margin-bottom:2rem; overflow:hidden;
    background:
        radial-gradient(circle at 15% 20%, rgba(124,92,255,0.18), transparent 45%),
        radial-gradient(circle at 85% 80%, rgba(203,166,247,0.16), transparent 50%),
        linear-gradient(180deg,#0a0a0a,#000);
}
.hero-grid{ display:grid; grid-template-columns:1.1fr 0.9fr; gap:2.5rem; align-items:center; }
.hero-eyebrow{
    display:inline-flex; align-items:center; gap:7px; font-size:0.78rem; font-weight:600;
    color:var(--accent); background:rgba(203,166,247,0.1);
    border:1px solid rgba(203,166,247,0.25); padding:5px 14px; border-radius:999px;
    text-transform:uppercase; letter-spacing:0.08em; margin-bottom:1.25rem;
}
.hero h1{
    font-size:3.4rem; font-weight:900; line-height:1.05; letter-spacing:-0.03em;
    margin:0 0 1rem;
}
.hero h1 .grad{
    background:linear-gradient(120deg,var(--accent),var(--accent-2));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.hero p{ font-size:1.15rem; color:var(--muted); line-height:1.6; margin:0 0 2rem; max-width:460px; }

/* ── Before/after showcase (pure CSS) ── */
.showcase{ position:relative; border-radius:18px; overflow:hidden;
    border:1px solid var(--border-2); aspect-ratio:4/3;
    box-shadow:0 24px 60px rgba(124,92,255,0.18); }
.showcase .half{ position:absolute; top:0; bottom:0; width:50%; }
.showcase .bw{ left:0; background:linear-gradient(135deg,#2a2a2a,#777,#1a1a1a); filter:grayscale(1); }
.showcase .color{ right:0;
    background:linear-gradient(135deg,#7c5cff,#CBA6F7,#ff9a8b,#ffd56b); }
.showcase .divider{ position:absolute; top:0; bottom:0; left:50%; width:2px;
    background:var(--accent); box-shadow:0 0 14px var(--accent); transform:translateX(-50%); }
.showcase .handle{ position:absolute; top:50%; left:50%; width:38px; height:38px;
    transform:translate(-50%,-50%); border-radius:50%; background:#000;
    border:2px solid var(--accent); box-shadow:0 0 18px var(--accent);
    display:flex; align-items:center; justify-content:center; color:var(--accent); font-weight:700; }
.showcase .tag{ position:absolute; bottom:12px; font-size:0.7rem; font-weight:600;
    text-transform:uppercase; letter-spacing:0.08em; padding:4px 10px; border-radius:6px;
    background:rgba(0,0,0,0.6); backdrop-filter:blur(4px); }
.showcase .tag.l{ left:12px; color:var(--muted); }
.showcase .tag.r{ right:12px; color:var(--accent); }

/* ── Buttons ── */
.stButton > button{
    background:linear-gradient(120deg,var(--accent),var(--accent-2));
    color:#000; border:none; border-radius:10px; font-weight:700; font-size:0.95rem;
    padding:0.7rem 1.8rem; transition:all 0.18s;
    box-shadow:0 6px 24px rgba(124,92,255,0.35);
}
.stButton > button:hover{ transform:translateY(-1px);
    box-shadow:0 10px 32px rgba(124,92,255,0.55); color:#000; }

.stDownloadButton > button{
    background:transparent; color:var(--fg); border:1px solid var(--border-2);
    border-radius:10px; font-weight:600; font-size:0.9rem; padding:0.6rem 1.4rem;
    transition:all 0.15s;
}
.stDownloadButton > button:hover{ border-color:var(--accent); color:var(--accent);
    box-shadow:0 0 0 3px rgba(203,166,247,0.12); }

/* ── Tabs (pill, connected to workflow) ── */
.stTabs [data-baseweb="tab-list"]{ gap:8px; background:transparent; border:none;
    margin-bottom:1.75rem; }
.stTabs [data-baseweb="tab"]{
    border-radius:999px; padding:9px 22px; font-size:0.92rem; font-weight:600;
    color:var(--muted); background:var(--panel); border:1px solid var(--border); }
.stTabs [aria-selected="true"]{
    background:linear-gradient(120deg,var(--accent),var(--accent-2)) !important;
    color:#000 !important; border-color:transparent !important;
    box-shadow:0 4px 18px rgba(124,92,255,0.4); }

/* ── Cards ── */
.card-title{ font-size:1.35rem; font-weight:800; letter-spacing:-0.02em;
    margin:0 0 0.4rem; }
.card-sub{ font-size:0.95rem; color:var(--muted); margin-bottom:1.5rem; }

/* ── Big upload zone ── */
[data-testid="stFileUploader"]{
    background:radial-gradient(circle at 50% 0%, rgba(203,166,247,0.08), transparent 70%), var(--panel);
    border:2px dashed var(--border-2); border-radius:var(--radius);
    padding:2.5rem 2rem; transition:all 0.2s;
}
[data-testid="stFileUploader"]:hover{
    border-color:var(--accent);
    box-shadow:0 0 0 4px rgba(203,166,247,0.1), 0 0 40px rgba(124,92,255,0.15);
}
[data-testid="stFileUploader"] section{ background:transparent !important; border:none !important; }
[data-testid="stFileUploaderDropzone"]{ background:transparent !important; }

.upload-hint{ text-align:center; margin-bottom:1rem; }
.upload-hint .icon{
    width:72px; height:72px; margin:0 auto 1rem; border-radius:20px;
    background:linear-gradient(135deg,rgba(203,166,247,0.18),rgba(124,92,255,0.12));
    border:1px solid rgba(203,166,247,0.3);
    display:flex; align-items:center; justify-content:center;
    box-shadow:0 0 30px rgba(124,92,255,0.2);
}
.upload-hint .icon svg{ width:34px; height:34px; stroke:var(--accent); }
.upload-hint .title{ font-size:1.4rem; font-weight:800; letter-spacing:-0.02em; }
.upload-hint .desc{ font-size:0.95rem; color:var(--muted); margin-top:0.4rem; }

/* ── Metric strip ── */
.metric-grid{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:1.5rem; }
.metric-card{ background:var(--panel); border:1px solid var(--border-2);
    border-radius:14px; padding:1.3rem 1.4rem; position:relative; overflow:hidden; }
.metric-card::before{ content:""; position:absolute; top:0; left:0; width:3px; height:100%;
    background:linear-gradient(var(--accent),var(--accent-2)); }
.metric-label{ font-size:0.74rem; font-weight:600; color:var(--muted);
    text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.5rem; }
.metric-value{ font-size:1.9rem; font-weight:800; color:var(--fg); line-height:1; letter-spacing:-0.02em; }
.metric-value.accent{ color:var(--accent); }
.metric-sub{ font-size:0.72rem; color:var(--muted); margin-top:0.35rem; }

.image-panel-label{ font-size:0.82rem; font-weight:700; color:var(--muted);
    text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.7rem; }

.section-head{ font-size:1.5rem; font-weight:800; letter-spacing:-0.02em; margin:0.5rem 0 1rem; }

/* ── Sample preview before upload ── */
.sample-note{ text-align:center; font-size:0.85rem; color:var(--muted); margin:1.5rem 0 0.5rem; }

.status-pill{ display:inline-flex; align-items:center; gap:7px; background:var(--panel);
    border:1px solid var(--border-2); color:var(--accent); font-size:0.8rem; font-weight:600;
    padding:5px 14px; border-radius:999px; }
.status-pill .dot{ width:7px; height:7px; border-radius:50%; background:var(--accent);
    box-shadow:0 0 8px var(--accent); }

.batch-meta{ font-size:0.74rem; color:var(--muted); margin-top:0.5rem;
    display:flex; justify-content:space-between; }
.stSlider [role=slider]{ background:var(--accent); }
.stProgress > div > div{ background:linear-gradient(90deg,var(--accent),var(--accent-2)); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# NAVBAR
# ─────────────────────────────────────────────

st.markdown("""
<div class="nav">
  <div class="brand"><span class="mark"></span>Colorize<span class="accent">AI</span> Studio</div>
  <div class="nav-links">
    <a href="https://github.com/Ayo-aditya2466/B-W-image-Colourization" target="_blank">GitHub</a>
   
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <div class="hero-grid">
    <div>
      <span class="hero-eyebrow">● AI-Powered Colorization</span>
      <h1>Bring black-and-white photos <span class="grad">to life</span></h1>
      <p>ColorizeAI Studio uses a deep neural network trained on 1.3M images
         to add realistic, vivid color to your old black-and-white photos in seconds.</p>
    </div>
    <div class="showcase">
      <div class="half bw"></div>
      <div class="half color"></div>
      <div class="divider"></div>
      <div class="handle">⟷</div>
      <span class="tag l">Before</span>
      <span class="tag r">After</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Primary CTA + status under hero
cta1, cta2, cta3 = st.columns([1, 4])

with cta2:
    st.markdown('<div style="padding-top:0.55rem;"><span class="status-pill">'
                '<span class="dot"></span> Model Ready</span></div>',
                unsafe_allow_html=True)

st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

ss = st.session_state
ss.setdefault("history", [])
ss.setdefault("last_result", None)
ss.setdefault("last_original", None)
ss.setdefault("sample_request", None)
ss.setdefault("sample_name", None)
# ─────────────────────────────────────────────
# MODEL PATHS & LOADER  (unchanged)
# ─────────────────────────────────────────────

DIR      = os.path.dirname(os.path.abspath(__file__))
PROTOTXT = os.path.join(DIR, "model", "colorization_deploy_v2.prototxt")
POINTS   = os.path.join(DIR, "model", "pts_in_hull.npy")
MODEL    = os.path.join(DIR, "model", "colorization_release_v2.caffemodel")
SAMPLES_DIR = os.path.join(DIR, "images")

SAMPLES = [
    "building.jpg",
    "einstein.jpg",
    "nature.jpg",
    "rose.jpg",
    "tiger.jpg"
]


@st.cache_resource(show_spinner=False)
def load_model():
    for path, name in [(PROTOTXT, "prototxt"), (MODEL, "caffemodel"), (POINTS, "pts_in_hull.npy")]:
        if not os.path.exists(path):
            st.error(f"Missing model file: {name}. Place files in /model folder.")
            st.stop()
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS).transpose().reshape(2, 313, 1, 1)
    class8 = net.getLayerId("class8_ab")
    conv8  = net.getLayerId("conv8_313_rh")
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs  = [np.full([1, 313], 2.606, dtype="float32")]
    return net

# ─────────────────────────────────────────────
# CORE FUNCTIONS  (logic unchanged)
# ─────────────────────────────────────────────

def colorize_image(bgr_image, net):
    start     = time.time()
    scaled    = bgr_image.astype("float32") / 255.0
    lab       = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    L_full    = cv2.split(lab)[0]
    L_resized = cv2.resize(L_full, (224, 224)) - 50
    net.setInput(cv2.dnn.blobFromImage(L_resized))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (bgr_image.shape[1], bgr_image.shape[0]))
    colorized = np.concatenate((L_full[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    return colorized, round(time.time() - start, 2)


def colorfulness_score(bgr_image):
    rgb     = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype("float32")
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    rg      = R - G
    yb      = 0.5 * (R + G) - B
    score   = (np.sqrt(np.std(rg)**2 + np.std(yb)**2) +
               0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2))
    return round(float(score), 1)


def is_grayscale(bgr_image, threshold=10):
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean()) < threshold


def bgr_to_pil(bgr_image):
    return Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))


def pil_to_bytes(pil_image, fmt="PNG"):
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    return buf.getvalue()


def uploaded_to_bgr(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

class SampleFile:
    def __init__(self, name, data):
        self.name = name
        self._buf = io.BytesIO(data)
        self.size = len(data)

    def read(self):
        return self._buf.getvalue()

    def seek(self, pos):
        return self._buf.seek(pos)

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

def detect_faces(bgr_image):

    gray = cv2.cvtColor(
        bgr_image,
        cv2.COLOR_BGR2GRAY
    )

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    output = bgr_image.copy()
    face_crops = []

    for (x, y, w, h) in faces:

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (203, 166, 247),
            3
        )
        crop = bgr_image[
            y:y+h,
            x:x+w
        ]

        if crop.size > 0:
            face_crops.append(crop)
            
    return output, len(faces), face_crops

def score_label(score):
    if score >= 60: return "High"
    if score >= 35: return "Medium"
    return "Low"


def metric_grid(items):
    html = '<div class="metric-grid">'
    for label, value, sub, accent in items:
        cls = "metric-value accent" if accent else "metric-value"
        sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
        html += (f'<div class="metric-card"><div class="metric-label">{label}</div>'
                 f'<div class="{cls}">{value}</div>{sub_html}</div>')
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


UPLOAD_ICON = """
<svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/>
<line x1="12" y1="3" x2="12" y2="15"/></svg>"""

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

with st.spinner("Loading model..."):
    net = load_model()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab_single, tab_batch, tab_enhance, tab_analytics = st.tabs(
    ["Single Image", "Batch Processing", "Enhance", "Analytics"]
)

# ════════════════════════════════════════════
# SINGLE IMAGE
# ════════════════════════════════════════════

with tab_single:
    st.markdown('<div class="card-title">Colorize an image</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-sub">Drag a black-and-white photo into the workspace '
                'below to generate a full-color result instantly.</div>', unsafe_allow_html=True)

    # ── Sample Gallery ──
    st.markdown('<div class="image-panel-label" style="margin-bottom:1rem;">'
                'Try a sample</div>', unsafe_allow_html=True)
    gal_cols = st.columns(len(SAMPLES), gap="small")
    for col, fname in zip(gal_cols, SAMPLES):
        fpath = os.path.join(SAMPLES_DIR, fname)
        with col:
            if os.path.exists(fpath):
                st.image(fpath, width="stretch")
                if st.button("Use Sample", key=f"sample_{fname}", width="stretch"):
                    with open(fpath, "rb") as fh:
                        ss.sample_request = fh.read()
                    ss.sample_name = fname
                    st.rerun()
            else:
                st.markdown(f'<div style="color:var(--muted);font-size:0.78rem;'
                            f'text-align:center;">{fname} missing</div>',
                            unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="upload-hint">
        <div class="icon">{UPLOAD_ICON}</div>
        <div class="title">Drop Image Here</div>
        <div class="desc">or click below to browse · JPG, JPEG, PNG</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload", type=["jpg", "jpeg", "png"],
                                key="single", label_visibility="collapsed")

    # Resolve active image source: user upload OR selected sample
    if uploaded is None and ss.sample_request is not None:
        uploaded = SampleFile(ss.sample_name, ss.sample_request)
    ss.sample_request = None  # one-shot: future reruns prefer real uploads

    if not uploaded:
        # Sample before/after preview — demonstrate expected results
        st.markdown('<div class="sample-note">Example result — your photos will look like this</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="showcase" style="max-width:760px; margin:0.5rem auto 0;">
          <div class="half bw"></div><div class="half color"></div>
          <div class="divider"></div><div class="handle">⟷</div>
          <span class="tag l">Before</span><span class="tag r">After</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        uploaded.seek(0)
        bgr = uploaded_to_bgr(uploaded)
        if bgr is None:
            st.error("Could not read image. Please try a different file.")
            st.stop()

        if not is_grayscale(bgr):

            st.warning(
                "This image appears to already be in colour."
            )

            choice = st.radio(
                "How would you like to continue?",
                [
                    "Use Original Image",
                    "Convert To Black & White First"
                ],
                horizontal=True
            )

            if choice == "Convert To Black & White First":
                gray = cv2.cvtColor(
                    bgr,
                    cv2.COLOR_BGR2GRAY
                )

                bgr = cv2.cvtColor(
                    gray,
                    cv2.COLOR_GRAY2BGR
                )

        
        with st.spinner("Colorizing with AI..."):
            result, elapsed = colorize_image(bgr, net)

        score = colorfulness_score(result)
        h, w  = bgr.shape[:2]
        fsize = round(uploaded.size / 1024, 1)

        ss.last_result   = result
        ss.last_original = bgr
        ss.history.append({"name": uploaded.name, "time": elapsed,
                           "score": score, "resolution": f"{w}x{h}"})

        metric_grid([
            ("Processing Time", f"{elapsed}s", None, False),
            ("Colorfulness", f"{score}", score_label(score), True),
            ("Resolution", f"{w}×{h}", None, False),
            ("File Size", f"{fsize} KB", None, False),
        ])

        
        
        st.markdown('<div id="results-anchor"></div>',unsafe_allow_html=True)

        st.markdown('<div class="section-head">Original vs Colorized</div>', unsafe_allow_html=True)
        if HAS_SLIDER:
            image_comparison(img1=bgr_to_pil(bgr), img2=bgr_to_pil(result),
                             label1="Original", label2="Colorized", width=1180)
        else:
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                st.markdown('<div class="image-panel-label">Original</div>', unsafe_allow_html=True)
                st.image(bgr_to_pil(bgr), width="stretch")
            with c2:
                st.markdown('<div class="image-panel-label">Colorized</div>', unsafe_allow_html=True)
                st.image(bgr_to_pil(result), width="stretch")

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        

        face_img, face_count, face_crops = detect_faces(
             result
        )
        if face_count > 0:

            st.markdown(
                '<div class="section-head">Face Detection</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"Detected **{face_count}** face(s)"
            )

            fd1, fd2 = st.columns([2, 1])

            with fd1:
                st.image(
                    bgr_to_pil(face_img),
                    width="stretch"
                )

            with fd2:
                st.metric(
                    "Faces Detected",
                    face_count
                )
        if face_crops:

            st.markdown(
                "#### Face Crops"
            )

            crop_cols = st.columns(
                min(3, len(face_crops))
            )

            for i, crop in enumerate(face_crops):

                with crop_cols[i % len(crop_cols)]:

                    st.image(
                        bgr_to_pil(crop),
                        width="stretch"
                    )

                    st.caption(
                        f"Face {i+1}"
                    )

        d1, d2 = st.columns([1, 3])
        with d1:
            st.download_button("Download Colorized Image",
                               data=pil_to_bytes(bgr_to_pil(result)),
                               file_name=f"colorized_{uploaded.name}",
                               mime="image/png", width='stretch')

# ════════════════════════════════════════════
# BATCH PROCESSING
# ════════════════════════════════════════════

with tab_batch:
    st.markdown('<div class="card-title">Batch processing</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-sub">Upload multiple black-and-white images. '
                'All results are packaged into a single ZIP for download.</div>',
                unsafe_allow_html=True)

    batch_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"],
                                   accept_multiple_files=True, key="batch",
                                   label_visibility="collapsed")

    if batch_files:
        st.markdown(f'<div style="color:var(--muted);font-size:0.9rem;margin:0.75rem 0 1rem;">'
                    f'{len(batch_files)} image{"s" if len(batch_files) > 1 else ""} selected</div>',
                    unsafe_allow_html=True)

        b1, b2 = st.columns([1, 3])
        with b1:
            run_batch = st.button("Colorize All",  use_container_width=True, key="run_batch")

        if run_batch:
            progress_bar = st.progress(0)
            status_text  = st.empty()
            results      = []
            zip_buffer   = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for i, f in enumerate(batch_files):
                    status_text.markdown(
                        f'<div style="color:var(--muted);font-size:0.85rem;">'
                        f'Processing {f.name} — {i+1} of {len(batch_files)}</div>',
                        unsafe_allow_html=True)
                    progress_bar.progress((i + 1) / len(batch_files))
                    f.seek(0)
                    bgr = uploaded_to_bgr(f)
                    if bgr is None:
                        continue
                    result, elapsed = colorize_image(bgr, net)
                    score = colorfulness_score(result)
                    results.append((f.name, result, elapsed, score))
                    ss.history.append({"name": f.name, "time": elapsed, "score": score,
                                       "resolution": f"{bgr.shape[1]}x{bgr.shape[0]}"})
                    zf.writestr(f"colorized_{f.name}", pil_to_bytes(bgr_to_pil(result)))

            status_text.empty(); 

            progress_bar.progress(1.0)
            if results:
                avg_time  = round(sum(r[2] for r in results) / len(results), 2)
                avg_score = round(sum(r[3] for r in results) / len(results), 1)
                best      = max(results, key=lambda x: x[3])
                metric_grid([
                    ("Images Processed", f"{len(results)}", None, True),
                    ("Avg Time", f"{avg_time}s", None, False),
                    ("Avg Colorfulness", f"{avg_score}", None, False),
                    ("Best Score", f"{best[3]}",
                     best[0][:14] + ("..." if len(best[0]) > 14 else ""), False),
                ])

        

                st.markdown(
                    '<div class="section-head">Results</div>',
                    unsafe_allow_html=True
                )

                cols = st.columns(3, gap="medium")

                for idx, (name, result_img, elapsed, score) in enumerate(results):

                    with cols[idx % 3]:

                        st.image(
                            bgr_to_pil(result_img),
                            width="stretch"
                        )

                        st.markdown(
                            f'<div class="batch-meta">'
                            f'<span>{name[:18]}{"..." if len(name) > 18 else ""}</span>'
                            f'<span>{elapsed}s · {score}</span></div>',
                            unsafe_allow_html=True
                        )

                st.markdown(
                    "<div style='height:1rem'></div>",
                    unsafe_allow_html=True
                )

                z1, z2 = st.columns([1, 3])

                with z1:

                    st.download_button(
                        f"Download {len(results)} Images as ZIP",
                        data=zip_buffer.getvalue(),
                        file_name="colorized_batch.zip",
                        mime="application/zip",
                        width='stretch'
                    )

# ════════════════════════════════════════════
# ENHANCE
# ════════════════════════════════════════════

with tab_enhance:
    st.markdown('<div class="card-title">Enhance result</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-sub">Fine-tune your latest colorized image '
                'with brightness, contrast, saturation, and sharpness.</div>',
                unsafe_allow_html=True)

    if ss.last_result is None:
        st.info("Colorize an image in the Single Image tab first — enhancements apply to your latest result.")
    else:
        base = bgr_to_pil(ss.last_result)
        ctrl, preview = st.columns([1, 2], gap="large")
        with ctrl:
            brightness = st.slider("Brightness", 0.5, 1.5, 1.0, 0.05)
            contrast   = st.slider("Contrast",   0.5, 1.5, 1.0, 0.05)
            saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.05)
            sharpness  = st.slider("Sharpness",  0.0, 2.0, 1.0, 0.05)

        enhanced = base
        for enh, val in [(ImageEnhance.Brightness, brightness), (ImageEnhance.Contrast, contrast),
                         (ImageEnhance.Color, saturation), (ImageEnhance.Sharpness, sharpness)]:
            enhanced = enh(enhanced).enhance(val)

        with preview:
            st.markdown('<div class="image-panel-label">Live Preview</div>', unsafe_allow_html=True)
            st.image(enhanced, width="stretch")

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        e1, e2 = st.columns([1, 3])
        with e1:
            st.download_button("Download Enhanced Image", data=pil_to_bytes(enhanced),
                               file_name="enhanced.png", mime="image/png",
                               width='stretch')

# ════════════════════════════════════════════
# ANALYTICS
# ════════════════════════════════════════════

with tab_analytics:
    st.markdown('<div class="card-title">Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-sub">Session performance and processing history.</div>',
                unsafe_allow_html=True)

    if not ss.history:
        st.info("No session activity yet — colorize images to populate analytics.")
    else:
        df = pd.DataFrame(ss.history)
        metric_grid([
            ("Total Processed", f"{len(df)}", None, True),
            ("Avg Time", f"{df['time'].mean():.2f}s", None, False),
            ("Avg Colorfulness", f"{df['score'].mean():.1f}", None, False),
            ("Peak Score", f"{df['score'].max():.1f}", None, False),
        ])

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<div class="image-panel-label">Processing Time</div>', unsafe_allow_html=True)
            st.altair_chart(
                alt.Chart(df.reset_index()).mark_line(point=True, color="#CBA6F7")
                .encode(x=alt.X("index:Q", title="Run"), y=alt.Y("time:Q", title="Seconds"))
                .properties(height=240).configure_view(strokeWidth=0)
                .configure_axis(grid=False, labelColor="#9a9a9a", titleColor="#9a9a9a")
                .configure(background="#000000"),
                 use_container_width=True)
        with c2:
            st.markdown('<div class="image-panel-label">Colorfulness</div>', unsafe_allow_html=True)
            st.altair_chart(
                alt.Chart(df.reset_index()).mark_bar(color="#CBA6F7")
                .encode(x=alt.X("index:Q", title="Run"), y=alt.Y("score:Q", title="Score"))
                .properties(height=240).configure_view(strokeWidth=0)
                .configure_axis(grid=False, labelColor="#9a9a9a", titleColor="#9a9a9a")
                .configure(background="#000000"),
                use_container_width=True)

        st.markdown('<div class="section-head">Session History</div>', unsafe_allow_html=True)

        st.dataframe(df, width="stretch", hide_index=True)
