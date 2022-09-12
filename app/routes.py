from flask import render_template
from app import app
from flask import json, render_template, request, redirect
from flask.json import jsonify
import numpy as np
import cv2
from app.ocr.detection import Detector
from app.ocr import extract_infos
from flask.helpers import url_for
import os

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_extract", methods=["POST"])
def get_extract():
    pass

THRESHOLD = 0.3
imageSize = 512
seq2seq_url = 'app/ocr/vietocr/seq2seqocr.pth'
targetSize = { 'w': imageSize, 'h': imageSize }

def url_to_image(url):
    """
    Read image from url
    """
    # resp = urllib.request.urlopen(url)
    # image = np.asarray(bytearray(resp.read()), dtype="uint8")
    # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.imread(url)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

@app.route('/api/v1/ocr', methods=['GET', 'POST'])
def api_ocr():
    # data = json.loads(request.data)
    # img_url = data['img_url']
    # corner_url = data['corner_url']
    # text_url = data['text_url']
    image = url_to_image('C:/Users/tiend/Downloads/facenet_embedding/datle1.JPG')
    image = np.array(image)
    targetSize['h'] = image.shape[0]
    targetSize['w'] = image.shape[1]

    detect_corner = Detector('C:/Users/tiend/Documents/ocr_id_card/app/models/ctc_chip/corner/saved_model', threshold=THRESHOLD, targetSize=targetSize)
    detect_text = Detector('C:/Users/tiend/Documents/ocr_id_card/app/models/ctc_chip/text/saved_model', threshold=THRESHOLD, targetSize=targetSize)
    # try:  
    corner_img = detect_corner.detect_corner(image, THRESHOLD, targetSize)
    # except:
    #     return jsonify({'errorCode': 1, 'errorMessage': 'can not detect id card. Id card need contain 4 corners'})

    targetSize['h'] = corner_img.shape[0]
    targetSize['w'] = corner_img.shape[1]

    textt = detect_text.detect_text(corner_img, THRESHOLD, targetSize)

    test = extract_infos.ocr(seq2seq_url)

    data = test.OCR(corner_img, textt)
    
    return jsonify(data)