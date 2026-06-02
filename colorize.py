"""
Black and White Image Colorization using OpenCV + Tkinter GUI
"""

# Import Libraries
import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# ---------------- MODEL PATHS ---------------- #

DIR = os.path.dirname(os.path.abspath(__file__))
PROTOTXT = os.path.join(DIR, r"model/colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"model/pts_in_hull.npy")
MODEL = os.path.join(DIR, r"model/colorization_release_v2.caffemodel")

# ---------------- LOAD MODEL ---------------- #

print("Loading Model...")

net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")

pts = pts.transpose().reshape(2, 313, 1, 1)

net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

print("Model Loaded Successfully!")

# ---------------- GUI WINDOW ---------------- #

root = tk.Tk()
root.title("Black & White Image Colorization")
root.geometry("1000x600")
root.configure(bg="lightblue")

# Global Variables
original_img = None
colorized_img = None

# ---------------- FUNCTIONS ---------------- #

def upload_image():
    global original_img, colorized_img

    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        return

    # Read image
    image = cv2.imread(file_path)

    if image is None:
        messagebox.showerror("Error", "Unable to load image!")
        return

    original_img = image

    # Convert BGR to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize for display
    rgb = cv2.resize(rgb, (400, 400))

    img = Image.fromarray(rgb)
    img = ImageTk.PhotoImage(img)

    original_label.config(image=img)
    original_label.image = img

    # Perform Colorization
    colorized_img = colorize_image(original_img)

    # Display Colorized Image
    display_colorized(colorized_img)


def colorize_image(image):

    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))

    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))

    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]

    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)

    colorized = np.clip(colorized, 0, 1)

    colorized = (255 * colorized).astype("uint8")

    return colorized


def display_colorized(image):

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rgb = cv2.resize(rgb, (400, 400))

    img = Image.fromarray(rgb)
    img = ImageTk.PhotoImage(img)

    colorized_label.config(image=img)
    colorized_label.image = img


def save_image():

    global colorized_img

    if colorized_img is None:
        messagebox.showwarning("Warning", "No image to save!")
        return

    save_path = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=[("JPEG File", "*.jpg"),
                   ("PNG File", "*.png")]
    )

    if save_path:
        cv2.imwrite(save_path, colorized_img)
        messagebox.showinfo("Success", "Image Saved Successfully!")


# ---------------- HEADING ---------------- #

title = tk.Label(
    root,
    text="Black & White Image Colorization",
    font=("Arial", 22, "bold"),
    bg="lightblue",
    fg="darkblue"
)

title.pack(pady=20)

# ---------------- BUTTONS ---------------- #

upload_btn = tk.Button(
    root,
    text="Upload Image",
    font=("Arial", 14, "bold"),
    bg="green",
    fg="white",
    command=upload_image
)

upload_btn.pack(pady=10)

save_btn = tk.Button(
    root,
    text="Save Colorized Image",
    font=("Arial", 14, "bold"),
    bg="blue",
    fg="white",
    command=save_image
)

save_btn.pack(pady=10)

# ---------------- IMAGE FRAMES ---------------- #

frame = tk.Frame(root, bg="lightblue")
frame.pack(pady=20)

# Original Image Label
original_text = tk.Label(
    frame,
    text="Original Image",
    font=("Arial", 16, "bold"),
    bg="lightblue"
)

original_text.grid(row=0, column=0, padx=40)

original_label = tk.Label(frame, bg="white", width=400, height=400)
original_label.grid(row=1, column=0, padx=20)

# Colorized Image Label
colorized_text = tk.Label(
    frame,
    text="Colorized Image",
    font=("Arial", 16, "bold"),
    bg="lightblue"
)

colorized_text.grid(row=0, column=1, padx=40)

colorized_label = tk.Label(frame, bg="white", width=400, height=400)
colorized_label.grid(row=1, column=1, padx=20)

# ---------------- RUN GUI ---------------- #

root.mainloop()
