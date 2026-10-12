# VisionAI – Image Detection and Captioning App

VisionAI is a web application built using Flask and deep learning techniques to detect objects in images and generate descriptive information about their contents. Users can upload images, view detected objects, and explore image-related insights through a simple web interface.

## Features

- Upload and process images through a web interface.
- Detect common objects using a TensorFlow object detection model trained on the COCO dataset.
- Generate descriptive text based on image content.
- Display detected labels and image information.
- View image insights and metadata.
- Access dedicated pages for detection, captioning, insights, and resources.

## Technologies Used

- **Programming Language:** Python
- **Web Framework:** Flask
- **Deep Learning:** TensorFlow
- **Image Processing:** Pillow
- **Numerical Computing:** NumPy
- **Frontend:** HTML, CSS
- **Object Detection Model:** SSD MobileNet V1 trained on the COCO dataset

## Project Structure

```text
VisionAI/
├── app.py
├── cap.py
├── requirements.txt
├── readme.md
├── data/
├── utils/
├── templates/
├── static/
└── uploads/
```

- `app.py` – Main Flask application.
- `cap.py` – Captioning-related functionality.
- `requirements.txt` – Python dependencies.
- `data/` – Label map and related data files.
- `utils/` – Object detection utility functions.
- `templates/` – HTML templates for the web interface.
- `static/` – Frontend assets and styling.
- `uploads/` – Directory for uploaded images.

## Requirements

- Python 3.10 or 3.11, subject to dependency compatibility.
- pip package manager.
- Required Python packages listed in `requirements.txt`.
- SSD MobileNet V1 COCO object detection model.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Akkuakanksha/Automatic_Image_Captioning.git
cd Automatic_Image_Captioning
```

If the project files are inside a nested directory, navigate into that directory before continuing.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install protobuf==3.20.3
```

### 4. Download the object detection model

Download the SSD MobileNet V1 COCO model archive:

[Download SSD MobileNet V1 COCO Model](https://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz)

Extract the archive into the project root. Verify that the following file exists:

```text
ssd_mobilenet_v1_coco_11_06_2017/
└── frozen_inference_graph.pb
```

## Running the Application

Start the Flask application from the project directory:

```bash
python app.py
```

Open your browser and visit:

http://localhost:5000/

## Application Pages

- **Image Detection:** Upload images and view detected objects.
- **Captioning:** Access the image captioning interface.
- **Insights:** View image information and detected labels.
- **Resources:** Explore project information and use cases.

## Future Enhancements

- Improve the quality and accuracy of generated captions.
- Support additional object detection models.
- Add more detailed image analysis.
- Improve the user interface and user experience.
- Optimize performance for larger images.

