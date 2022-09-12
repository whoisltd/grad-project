import requests
import numpy as np
import tensorflow as tf
import time
from app.ocr.functions import align_image, process_output

class Detector(object):
    def __init__(self, url_model, threshold, targetSize):
        self.url_model = url_model
        self.threshold = threshold
        self.targetSize = targetSize
        self.detect_fn = self.load_model(self.url_model)

    def load_model(self, model_url):
        """
        Load model from url

        Parameters:
            model_url: string url of model
        Returns:
            model: model
            detect_fn: function to detect
        """
        start_time = time.time()
        model = tf.saved_model.load(model_url)
        detect_fn = model.signatures['serving_default']
        print('Load model time: ', time.time() - start_time)
        return detect_fn

    def detect_corner(self, img, threshold, targetSize):
        """
        Detect corner from img

        Parameters:
            img: image tensor
            url_model: string url of model
            threshold: float, threshold of confidence
            targetSize: int, target size of image
        Returns:
            corner: float tensor of shape [1, 4]
        """
        # image = np.expand_dims(img, axis=0)
        # image = tf.make_tensor_proto(image)
        input_tensor = tf.convert_to_tensor(img)
        input_tensor = input_tensor[tf.newaxis, ...]

        results = self.detect_fn(input_tensor)
        # results = get_grpc_predict(url_model, 'input_tensor', image)
        # results = results.outputs

        # payload = {'instances': [img.tolist()]}
        # res = requests.post(url_model, json=payload)
        # data= res.json()['predictions'][0]
        # results = process_output('corner', data, threshold, targetSize)
        print(results)
        results = process_output('corner', results, threshold, targetSize)
        
        crop_img = align_image(img, results)
        crop_img = np.array(crop_img)

        return crop_img
    
    def detect_text(self, img, threshold, targetSize):
        """
        Detect text from img

        Parameters:
            img: image tensor
            url_model: string url of model
            threshold: float, threshold of confidence
            targetSize: int, target size of image
        Returns:
            text: float tensor of shape [1, 4]
        """
        # image = np.expand_dims(img, axis=0)
        # image = tf.make_tensor_proto(image)
        input_tensor = tf.convert_to_tensor(img)
        input_tensor = input_tensor[tf.newaxis, ...]

        results = self.detect_fn(input_tensor)
        # results = get_grpc_predict(url_model, 'input_tensor', image)
        # results = results.outputs

        # payload = {'instances': [img.tolist()]}
        # res = requests.post(url_model, json=payload)
        # data= res.json()['predictions'][0]
        # results = process_output('text', data, threshold, targetSize)

        results = process_output('text', results, threshold, targetSize)

        return results