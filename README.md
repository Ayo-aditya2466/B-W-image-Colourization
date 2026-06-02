# Black & White Image Colorization

A Python-based desktop application that colorizes black-and-white images using OpenCV's Deep Neural Network (DNN) colorization model and a Tkinter graphical user interface.

## Features

* Upload black-and-white images
* Automatic image colorization using a pretrained deep learning model
* Side-by-side comparison of original and colorized images
* Save the colorized output
* Simple and user-friendly Tkinter GUI

## Project Structure

```text
colourize/
│
├── colorize.py
├── README.md
├── .gitignore
│
├── images/
│   ├── building.jpg
│   ├── einstein.jpg
│   ├── nature.jpg
│   ├── rose.jpg
│   └── tiger.jpg
│
└── model/
    ├── colorization_deploy_v2.prototxt
    ├── pts_in_hull.npy
    └── readme.md
```

## Requirements

* Python 3.x
* OpenCV
* NumPy
* Pillow
* Tkinter

Install dependencies:

```bash
pip install numpy opencv-python pillow
```

## Download Model File

The pretrained model file is larger than GitHub's 100 MB upload limit and is therefore not included in this repository.

Download the model file:

**Google Drive:**
https://drive.google.com/drive/folders/1xdCBF_WPKQMCaGFCaZYu83UgVfKJOO1n

Download:

```text
colorization_release_v2.caffemodel
```

Place the downloaded file inside:

```text
model/
```

Final model folder:

```text
model/
├── colorization_deploy_v2.prototxt
├── pts_in_hull.npy
├── colorization_release_v2.caffemodel
└── readme.md
```

## Running the Project

Navigate to the project directory:

```bash
cd colourize
```

Run:

```bash
python colorize.py
```

You should see:

```text
Loading Model...
Model Loaded Successfully!
```

The GUI window will open automatically.

## How to Use

1. Click **Upload Image**
2. Select a black-and-white image
3. Wait for the model to process the image
4. View the colorized result
5. Click **Save Colorized Image** to save the output

## Technologies Used

* Python
* OpenCV DNN
* NumPy
* Tkinter
* Pillow

## Author

Aditya

## License

This project is created for educational and learning purposes.
