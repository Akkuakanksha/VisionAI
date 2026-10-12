import os
import zipfile
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
# Legacy TensorFlow object-detection code expects TF 1.x APIs.
tf.compat.v1.disable_v2_behavior()
from collections import defaultdict

# Use the TensorFlow 1.x API names expected by the project.
Graph = tf.compat.v1.Graph
GraphDef = tf.compat.v1.GraphDef
Session = tf.compat.v1.Session
gfile = tf.compat.v1.gfile
from io import StringIO
from PIL import Image
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_TO_CKPT = os.path.join(BASE_DIR, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(BASE_DIR, 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

detection_graph = None
category_index = {}

if os.path.exists(PATH_TO_CKPT) and os.path.exists(PATH_TO_LABELS):
    detection_graph = Graph()
    with detection_graph.as_default():
      od_graph_def = GraphDef()
      with gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def detect_objects_in_image(image_path):
    if detection_graph is None or not category_index:
        return []

    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    with detection_graph.as_default():
        with Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

    scores = np.squeeze(scores)
    classes = np.squeeze(classes).astype(np.int32)
    result = []

    for class_id, score in zip(classes, scores):
        if float(score) < 0.25:
            continue
        label = category_index.get(int(class_id), {}).get('name')
        if label and label not in result:
            result.append(label)

    return result[:5]


def build_caption(labels):
    if not labels:
        return 'A scene with no clear object detected.'
    if len(labels) == 1:
        return f'A photo of a {labels[0]}.'
    if len(labels) == 2:
        return f'A photo of a {labels[0]} and a {labels[1]}.'
    if len(labels) == 3:
        return f'A photo of a {labels[0]}, {labels[1]}, and {labels[2]}.'
    return f'A photo containing {", ".join(labels[:3])}, and more.'


def build_analysis_points(labels):
    primary_label = labels[0].lower() if labels else 'object'

    if primary_label == 'dog':
        return [
            f'Detected object: {labels[0]}',
            'Description: Dogs are loyal, intelligent animals often recognized by their expressive faces, wagging tails, and social behavior.',
            'Behavior: They are commonly seen playing, running, resting, or interacting closely with people.',
            'Environment: Dogs are usually found in homes, parks, streets, and outdoor areas where they can move freely.',
            'Real-world use: They support companionship, security, assistance work, and family life in everyday environments.'
        ]

    if not labels:
        return [
            'Detected object: No clear object identified.',
            'Description: The model could not confidently recognize any major element in the image.',
            'Behavior: The scene may be too complex, blurry, or contain small details that are hard to classify.',
            'Environment: The image may represent a natural or abstract scene rather than a clear object-focused photo.',
            'Real-world use: This can help prompt users to upload a clearer image for more accurate detection.'
        ]

    return [
        f'Detected object: {labels[0]}',
        f'Description: {labels[0].capitalize()} is identified as a meaningful object in the image and can be described by its visual form and context.',
        'Behavior: The object appears as a prominent part of the scene and contributes to the overall visual meaning of the image.',
        'Environment: It likely comes from a common real-world setting where similar objects are frequently observed.',
        'Real-world use: This helps in scene understanding, object recognition, and image-based automation tasks.'
    ]


def get_image_details(file_path):
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            extension = os.path.splitext(file_path)[1].upper().replace('.', '') or 'IMAGE'
            size_kb = round(os.path.getsize(file_path) / 1024, 2)
            return {
                'filename': os.path.basename(file_path),
                'extension': extension,
                'size_kb': size_kb,
                'width': width,
                'height': height,
            }
    except Exception:
        return {
            'filename': os.path.basename(file_path),
            'extension': 'IMAGE',
            'size_kb': round(os.path.getsize(file_path) / 1024, 2),
            'width': 0,
            'height': 0,
        }


def ensure_runtime_assets():
    for folder_name in ('static', 'templates'):
        folder_path = os.path.join(BASE_DIR, folder_name)
        if not os.path.exists(folder_path):
            archive = os.path.join(BASE_DIR, f'{folder_name}.zip')
            if os.path.exists(archive):
                with zipfile.ZipFile(archive, 'r') as zf:
                    zf.extractall(BASE_DIR)

ensure_runtime_assets()

app = Flask(__name__)
bootstrap = Bootstrap(app)

app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/captioning', methods=['GET', 'POST'])
def captioning():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            labels = detect_objects_in_image(filepath)
            caption = build_caption(labels)
            return render_template('captioning.html',
                                   caption=caption,
                                   image_url=url_for('uploaded_file', filename=filename),
                                   uploaded=True)
        return render_template('captioning.html', error='Please upload a valid image file.', uploaded=False)

    return render_template('captioning.html', caption='', image_url='', uploaded=False)


@app.route('/insights', methods=['GET', 'POST'])
def insights():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            labels = detect_objects_in_image(filepath)
            caption = build_caption(labels)
            details = get_image_details(filepath)
            summary_points = build_analysis_points(labels)
            return render_template('insights.html',
                                   caption=caption,
                                   image_url=url_for('uploaded_file', filename=filename),
                                   labels=labels,
                                   details=details,
                                   summary_points=summary_points,
                                   uploaded=True)
        return render_template('insights.html', error='Please upload a valid image file.', uploaded=False)

    return render_template('insights.html', caption='', image_url='', labels=[], details={}, summary_points=[], uploaded=False)


@app.route('/resources')
def resources():
    return render_template('resources.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_file',
                                filename=filename))
    return 'Invalid file upload.', 400


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    if detection_graph is None or not category_index:
        return (
            'The object-detection model files are missing. Download the TensorFlow COCO model and place '
            'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb in the project root before uploading images.',
            500,
        )

    PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR,filename.format(i)) for i in range(1, 2) ]
    IMAGE_SIZE = (12, 8)

    with detection_graph.as_default():
        with Session(graph=detection_graph) as sess:
            for image_path in TEST_IMAGE_PATHS:
                image = Image.open(image_path)
                image_np = load_image_into_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                im = Image.fromarray(image_np)
                im.save('uploads/'+filename)

    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)
