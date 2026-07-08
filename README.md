# ColorizeAI Studio

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-DNN-5C3EE8?logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?logo=streamlit&logoColor=white)
![Model](https://img.shields.io/badge/Model-Zhang%20et%20al.%202016-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

> **Bring black-and-white photos back to life using deep learning.**
> ColorizeAI Studio colorizes images in under 0.5 seconds using a CNN
> trained on 1.3 million ImageNet images — with batch processing,
> face detection, enhancement controls, and live analytics.

## Live Demo
👉 **[colorizeai-studio.streamlit.app](https://your-app.streamlit.app)**

---

## Features

| Feature | Description |
|---------|-------------|
| **AI Colorization** | Zhang et al. 2016 CNN via OpenCV DNN in LAB color space |
| **Comparison Slider** | Drag to reveal before/after side by side |
| **Face Detection** | Haar Cascade detects and crops faces in colorized output |
| **Enhancement Controls** | Adjust brightness, contrast, saturation, sharpness post-colorization |
| **Batch Processing** | Colorize multiple images at once, download as ZIP |
| **Analytics Dashboard** | Session history, score trends, processing time charts |
| **Sample Gallery** | Try 5 built-in test images instantly — no upload needed |
| **Auto B&W Converter** | Upload a colour photo — app converts it before colorizing |

---

## How It Works

```
Input (B&W image)
      ↓
Convert to CIELAB color space
      ↓
Extract L (luminance) channel
      ↓
Resize L to 224×224 (model input)
      ↓
CNN predicts 313 ab color bins (Zhang et al. 2016)
      ↓
Resize ab output back to original resolution
      ↓
Recombine L + predicted ab → convert to RGB
      ↓
Output (colorized image)
```

---

## Model

| Attribute | Detail |
|-----------|--------|
| Architecture | 16-layer CNN (VGG-style encoder-decoder) |
| Paper | Zhang et al., "Colorful Image Colorization", ECCV 2016 |
| Training data | 1.3M ImageNet images |
| Color space | CIELAB |
| Output | 313 ab color class bins |
| Inference engine | OpenCV DNN module |
| Runtime | ~0.15–0.5s per image (CPU) |

> This project uses the pretrained Zhang et al. 2016 model.
> The neural network architecture and weights were created by
> Richard Zhang, Phillip Isola, and Alexei A. Efros at UC Berkeley.
> My contribution is the full application layer: pipeline, GUI,
> batch processing, face detection, analytics, and deployment.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10 |
| CV / DNN | OpenCV 4.9 |
| Arrays | NumPy 1.26 |
| Image processing | Pillow 10 |
| Web app | Streamlit 1.45 |
| Charts | Altair |
| Comparison slider | streamlit-image-comparison |
| Face detection | OpenCV Haar Cascade |

---

## Project Structure

```
colorizeai-studio/
├── app.py                  ← Streamlit web application
├── colorize.py             ← Desktop Tkinter application (local)
├── model/
│   ├── colorization_deploy_v2.prototxt
│   ├── pts_in_hull.npy
│   └── colorization_release_v2.caffemodel  ← NOT in repo (see setup)
├── images/                 ← 5 sample test images
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── packages.txt
├── .gitignore
└── README.md
```

---

## Local Setup

```bash
# 1. Clone
git clone https://github.com/Ayo-aditya2466/colorizeai-studio.git
cd colorizeai-studio

# 2. Install
pip install -r requirements.txt

# 3. Add model file to model/ folder (see README for download link)

# 4. Run web app
streamlit run app.py

# 5. Or run desktop app
python colorize.py
```

---

## Skills Demonstrated

**Computer Vision** — LAB color space pipeline, DNN inference,
image preprocessing/postprocessing, face detection, colorfulness metric

**Deep Learning** — Pretrained CNN inference, Zhang et al. 2016
architecture (313-class ab classification, class rebalancing)

**Python Engineering** — Session state, threading, file I/O,
ZIP generation, image format conversion, caching

**Web Application** — Streamlit multipage app, custom CSS design
system, cloud deployment

---

## Author

**Aditya Mhetre**

- GitHub: [Ayo-aditya2466](https://github.com/Ayo-aditya2466)
- LinkedIn: [[Add your LinkedIn URL](https://www.linkedin.com/in/adityamhetre24/)]

---

## License

MIT License — free to fork, adapt, and build on.

---

## Reference

Zhang, R., Isola, P., & Efros, A. A. (2016).
*Colorful Image Colorization.* ECCV 2016.
[Paper](https://arxiv.org/abs/1603.08511)
