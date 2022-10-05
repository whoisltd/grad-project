from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import cv2


class ocr:
    def __init__(self, url_model):

        config = Cfg.load_config_from_file('app/ocr/vietocr/base.yml')
        # load vgg transformer config
        vgg_config = Cfg.load_config_from_file('app/ocr/vietocr/vgg-seq2seq.yml')

        # update base config
        config.update(vgg_config)
        # config = Cfg.load_config_from_name('vgg_seq2seq')
        config['weights'] = url_model
        config['cnn']['pretrained'] = False
        config['device'] = 'cpu'
        config['predictor']['beamsearch'] = False
        self.predictor = Predictor(config)

    def OCR(self, img, data):
        """
        extract text from image

        Parameters
            img: image tensor
            data: dict
        Returns
            data: dict
        """
        boxes = list(data.values())
        labels = list(data.keys())
        # print(boxes)
        print(labels)
        dict = {'errorCode': 0, 'data': {
            'data': {}, 'confidence': {}}, 'errorMsg': 'success'}
        data_label = dict['data']
        count = 0
        for i in range(len(boxes)):
            boxes[i] = sorted(boxes[i], key=lambda k: [k[3]], reverse=True)
            box = boxes[i]
            for j in range(len(box)):
                text_img = img[int(box[j][1]):int(box[j][3]),
                               int(box[j][0]):int(box[j][2])]
                cv2.imwrite('text_img' + str(count) + '.png', text_img)
                count+=1

                if labels[i] in data_label['data']:
                    text, conf = self.predictor.predict(
                        Image.fromarray(text_img), True)
                    print(text + "\n")
                    data_label['data'][labels[i]] = text + " " + data_label['data'][labels[i]]
                    data_label['confidence'][labels[i]] = (
                        conf + data_label['confidence'][labels[i]])/2
                else:
                    text, conf = self.predictor.predict(
                        Image.fromarray(text_img), True)
                    print(text + "\n")
                    data_label['data'][labels[i]] = text
                    data_label['confidence'][labels[i]] = conf
        dict['data'] = data_label
        return dict
