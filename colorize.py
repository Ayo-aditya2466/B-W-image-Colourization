"""
Black and White Image Colorization
Tech: OpenCV DNN (Zhang et al. 2016), Tkinter, PIL
Author: Aditya Mhetre
"""

import numpy as np
import cv2
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# ─────────────────────────────────────────────
# MODEL PATHS
# ─────────────────────────────────────────────

DIR       = os.path.dirname(os.path.abspath(__file__))
PROTOTXT  = os.path.join(DIR, "model", "colorization_deploy_v2.prototxt")
POINTS    = os.path.join(DIR, "model", "pts_in_hull.npy")
MODEL     = os.path.join(DIR, "model", "colorization_release_v2.caffemodel")

DISPLAY_SIZE = 380   # single constant for display thumbnail size

# ─────────────────────────────────────────────
# MODEL LOADER
# FIX: moved into a function with error handling
#      instead of running at module level
# ─────────────────────────────────────────────

net = None

def load_model():
    """Load the Zhang 2016 colorization model. Returns True on success."""
    global net
    try:
        # FIX: check files exist before loading — gives friendly error
        for path, label in [(PROTOTXT, "prototxt"), (MODEL, "caffemodel"), (POINTS, "pts_in_hull.npy")]:
            if not os.path.exists(path):
                messagebox.showerror(
                    "Model Missing",
                    f"Could not find: {label}\n\nPlace model files in the /model folder.\n"
                    f"Download link in README.md"
                )
                return False

        _net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
        pts  = np.load(POINTS)

        class8 = _net.getLayerId("class8_ab")
        conv8  = _net.getLayerId("conv8_313_rh")

        pts = pts.transpose().reshape(2, 313, 1, 1)
        _net.getLayer(class8).blobs = [pts.astype("float32")]
        _net.getLayer(conv8).blobs  = [np.full([1, 313], 2.606, dtype="float32")]

        net = _net
        return True

    except Exception as e:
        messagebox.showerror("Model Error", f"Failed to load model:\n{e}")
        return False

# ─────────────────────────────────────────────
# COLORIZE CORE
# FIX: cleaner L-channel handling, added timing
# ─────────────────────────────────────────────

def colorize_image(bgr_image):
    """
    Colorize a BGR image using Zhang et al. 2016.
    Returns (colorized_bgr, elapsed_seconds).
    """
    import time
    start = time.time()

    scaled = bgr_image.astype("float32") / 255.0
    lab    = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # FIX: extract L from original THEN resize — clearer logic
    L_full = cv2.split(lab)[0]

    # Resize L to model input size
    L_resized = cv2.resize(L_full, (224, 224)) - 50

    net.setInput(cv2.dnn.blobFromImage(L_resized))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize ab back to original image size
    ab = cv2.resize(ab, (bgr_image.shape[1], bgr_image.shape[0]))

    # Recombine L (original size) + predicted ab
    colorized = np.concatenate((L_full[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    elapsed = round(time.time() - start, 2)
    return colorized, elapsed

# ─────────────────────────────────────────────
# COLORFULNESS SCORE
# NEW: gives a quality metric on the output
# ─────────────────────────────────────────────

def colorfulness_score(bgr_image):
    """
    Hasler & Susstrunk (2003) colorfulness metric.
    Higher = more colorful output.
    """
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype("float32")
    R, G, B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    score = (np.sqrt(np.std(rg)**2 + np.std(yb)**2) +
             0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2))
    return round(float(score), 1)

# ─────────────────────────────────────────────
# ASPECT-RATIO AWARE RESIZE
# FIX: replaces fixed 400x400 squish
# ─────────────────────────────────────────────

def fit_image(bgr_image, size=DISPLAY_SIZE):
    """Resize image to fit inside size×size box, preserving aspect ratio."""
    h, w = bgr_image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(bgr_image, (new_w, new_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))

# ─────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────

root = tk.Tk()
root.title("B&W Image Colorizer — Aditya Mhetre")
root.geometry("1050x680")
root.configure(bg="#1e1e2e")
root.resizable(False, False)

# State
original_img   = None
colorized_img  = None

# ── Helpers ──────────────────────────────────

def set_status(text, color="#a6e3a1"):
    status_label.config(text=text, fg=color)
    root.update_idletasks()

def update_stats(elapsed, score):
    time_label.config(text=f"⏱  {elapsed}s")
    score_label.config(text=f"🎨  Score: {score}")

# ── Upload ────────────────────────────────────

def upload_image():
    global original_img, colorized_img

    if net is None:
        messagebox.showerror("Error", "Model not loaded.")
        return

    path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not path:
        return

    image = cv2.imread(path)
    if image is None:
        messagebox.showerror("Error", "Could not read image.")
        return

    original_img  = image
    colorized_img = None

    # FIX — store PhotoImage in a variable FIRST, then assign both
    photo = fit_image(original_img)
    original_label.config(image=photo, text="")
    original_label.image = photo   # same object — no garbage collection

    colorized_label.config(image="", text="Processing...", fg="#cdd6f4")
    set_status("Colorizing...", "#f9e2af")
    upload_btn.config(state="disabled")

    threading.Thread(target=_colorize_worker, daemon=True).start()
def _colorize_worker():
    """Background thread — does inference, then schedules GUI update."""
    global colorized_img
    result, elapsed = colorize_image(original_img)
    colorized_img   = result
    # Schedule GUI update back on main thread
    root.after(0, lambda: _colorize_done(elapsed))

def _colorize_done(elapsed):
    """Called on main thread after colorization finishes."""
    score = colorfulness_score(colorized_img)

    photo = fit_image(colorized_img)
    colorized_label.config(image=photo, text="")
    colorized_label.image = photo

    update_stats(elapsed, score)
    set_status(f"✅  Done in {elapsed}s   |   Colorfulness score: {score}")
    upload_btn.config(state="normal")

# ── Save ─────────────────────────────────────

def save_image():
    if colorized_img is None:
        messagebox.showwarning("Warning", "Colorize an image first.")
        return

    path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG File", "*.png"), ("JPEG File", "*.jpg")]
    )
    if path:
        cv2.imwrite(path, colorized_img)
        messagebox.showinfo("Saved", f"Image saved to:\n{path}")
        set_status(f"💾  Saved → {os.path.basename(path)}")

# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────

# Title
tk.Label(
    root, text="🎨  B&W Image Colorizer",
    font=("Arial", 22, "bold"),
    bg="#1e1e2e", fg="#cba6f7"
).pack(pady=(18, 4))

tk.Label(
    root, text="Powered by Zhang et al. 2016 · OpenCV DNN",
    font=("Arial", 10),
    bg="#1e1e2e", fg="#6c7086"
).pack()

# Stats bar
stats_frame = tk.Frame(root, bg="#1e1e2e")
stats_frame.pack(pady=8)

time_label  = tk.Label(stats_frame, text="⏱  —",
    font=("Arial", 11), bg="#1e1e2e", fg="#89dceb")
time_label.pack(side="left", padx=20)

score_label = tk.Label(stats_frame, text="🎨  Score: —",
    font=("Arial", 11), bg="#1e1e2e", fg="#89dceb")
score_label.pack(side="left", padx=20)

# Buttons — side by side
btn_frame = tk.Frame(root, bg="#1e1e2e")
btn_frame.pack(pady=8)

upload_btn = tk.Button(
    btn_frame, text="📂  Upload Image",
    font=("Arial", 13, "bold"),
    bg="#a6e3a1", fg="#1e1e2e", relief="flat",
    padx=18, pady=6,
    command=upload_image
)
upload_btn.pack(side="left", padx=12)

save_btn = tk.Button(
    btn_frame, text="💾  Save Result",
    font=("Arial", 13, "bold"),
    bg="#89b4fa", fg="#1e1e2e", relief="flat",
    padx=18, pady=6,
    command=save_image
)
save_btn.pack(side="left", padx=12)

# Image panels
panels = tk.Frame(root, bg="#1e1e2e")
panels.pack(pady=12)

for col, label_text in enumerate(["Original", "Colorized"]):
    tk.Label(
        panels, text=label_text,
        font=("Arial", 13, "bold"),
        bg="#1e1e2e", fg="#cdd6f4"
    ).grid(row=0, column=col, padx=40, pady=(0, 6))

original_label = tk.Label(
    panels, bg="#313244",
    width=DISPLAY_SIZE, height=DISPLAY_SIZE,
    text="No image loaded", fg="#6c7086",
    font=("Arial", 11)
)
original_label.grid(row=1, column=0, padx=30)

colorized_label = tk.Label(
    panels, bg="#313244",
    width=DISPLAY_SIZE, height=DISPLAY_SIZE,
    text="Colorized image\nwill appear here", fg="#6c7086",
    font=("Arial", 11)
)
colorized_label.grid(row=1, column=1, padx=30)

# Status bar
status_label = tk.Label(
    root, text="Ready — upload a B&W image to begin",
    font=("Arial", 10),
    bg="#1e1e2e", fg="#a6e3a1"
)
status_label.pack(pady=(10, 4))

# ─────────────────────────────────────────────
# STARTUP — load model after window is ready
# FIX: model loads after GUI opens, not before
# ─────────────────────────────────────────────

def startup():
    set_status("Loading model...", "#f9e2af")
    success = load_model()
    if success:
        set_status("✅  Model loaded — ready to colorize")
    else:
        set_status("❌  Model failed to load — check /model folder", "#f38ba8")

root.after(100, startup)  # run after mainloop starts
root.mainloop()