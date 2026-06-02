# Black & White Image Colorization

## Project Overview

This project automatically converts black-and-white (grayscale) images into realistic color images using Deep Learning techniques. The implementation utilizes OpenCV's DNN module along with a pre-trained Caffe colorization model to predict color information for grayscale images.

The model has been trained on a large dataset of color images and can generate visually appealing colorized outputs for historical photographs and grayscale images.

---

## Features

- Automatic grayscale image colorization
- Deep Learning-based color prediction
- Uses pre-trained Caffe colorization model
- Supports multiple image formats
- Simple and efficient implementation
- High-quality colorized outputs

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Pillow (PIL)
- Deep Neural Networks (DNN)
- Caffe Model

---

## Project Structure

```
B-W-image-Colourization/
│
├── images/
│   ├── building.jpg
│   ├── einstein.jpg
│   ├── nature.jpg
│   ├── rose.jpg
│   └── tiger.jpg
│
├── model/
│   ├── colorization_deploy_v2.prototxt
│   ├── colorization_release_v2.caffemodel
│   └── pts_in_hull.npy
│
├── colorize.py
├── README.md
└── .gitignore
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/Ayo-aditya2466/B-W-image-Colourization.git
```

### Navigate to Project Directory

```bash
cd B-W-image-Colourization
```

### Install Required Libraries

```bash
pip install opencv-python numpy pillow
```

---

## Required Model Files

The project uses the following pre-trained Caffe model files:

- `colorization_deploy_v2.prototxt`
- `colorization_release_v2.caffemodel`
- `pts_in_hull.npy`

These files are stored in the `model` directory and are required for the colorization process.

---

## Running the Project

```bash
python colorize.py
```

---

## Working Methodology

1. Load the grayscale image.
2. Load the pre-trained Caffe colorization model.
3. Extract luminance information from the image.
4. Pass image data through the Deep Neural Network.
5. Predict color channels.
6. Combine luminance and predicted color information.
7. Generate the final colorized image.

---

## Applications

- Restoration of historical photographs
- Image enhancement
- Digital media processing
- Computer vision research
- AI-assisted photo editing

---

## Future Enhancements

- Real-time image colorization
- Batch image processing
- Web-based user interface
- Support for video colorization
- Higher-resolution output generation

---

## Author

Aditya Mhetre

GitHub: https://github.com/Ayo-aditya2466
