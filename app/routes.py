from flask import render_template
from app import app
from flask import json, render_template, jsonify, request
from flask.json import jsonify
import numpy as np
import cv2
from app.ocr.detection import Detector
from app.ocr import extract_infos
from app.ocr.functions import load_labels_map, stringToRGB
import os
import tensorflow as tf

THRESHOLD = 0.3
imageSize = 512
seq2seq_url = 'app/ocr/vietocr/seq2seqocr.pth'
targetSize = { 'w': imageSize, 'h': imageSize }
label_map_corner = load_labels_map('app/ocr/label_map/corner.pbtxt')
label_map_text = load_labels_map('app/ocr/label_map/text.pbtxt')
label_map_text_chip = load_labels_map('app/ocr/label_map/text_chip.pbtxt')
#load keras model
classifi = tf.keras.models.load_model('app/models/classification/classification.h5')

corner_chip = Detector('app/models/ctc_chip/corner/saved_model', THRESHOLD, targetSize, label_map_corner)
text_chip = Detector('app/models/ctc_chip/text/saved_model', THRESHOLD, targetSize, label_map_text_chip)

corner = Detector('app/models/id_card/corner/saved_model', THRESHOLD, targetSize, label_map_corner)
text = Detector('app/models/id_card/text/saved_model', THRESHOLD, targetSize, label_map_text)

ocr_model = extract_infos.ocr(seq2seq_url)

detect_corner = None
detect_text = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/api/v1/ocr', methods=['GET', 'POST'])
def api_ocr():
    # image = url_to_image('C:/Users/tiend/Downloads/image_f83d36fc-f415-41e1-999d-814d09528e9e.jpg')
    
    # image = np.array(image)
    image = None
    if request.method == 'POST':
        data = json.loads(request.data)
        image = stringToRGB(data['img'])
    
    #classifi
    img = cv2.resize(image, (150, 150))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    pred = classifi.predict(img)

    targetSize['h'] = image.shape[0]
    targetSize['w'] = image.shape[1]

    
    if pred[0][0] < 0.5:
        detect_corner = corner_chip
        detect_text = text_chip
    else:
        detect_corner = corner
        detect_text = text

    try:  
        corner_img = detect_corner.detect_corner(image, THRESHOLD, targetSize)
        cv2.imwrite('app/ocr/corner.jpg', corner_img)
    except:
        return jsonify({'errorCode': 1, 'errorMessage': 'can not detect id card. Id card need contain 4 corners'})

    targetSize['h'] = corner_img.shape[0]
    targetSize['w'] = corner_img.shape[1]

    textt = detect_text.detect_text(corner_img, THRESHOLD, targetSize)

    data = ocr_model.OCR(corner_img, textt)
    print(data)
    return jsonify(data)


# def url_to_image(url):
#     """
#     Read image from url
#     """
#     # resp = urllib.request.urlopen(url)
#     # image = np.asarray(bytearray(resp.read()), dtype="uint8")
#     # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#     image = cv2.imread(url)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     return image
