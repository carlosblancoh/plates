import os
import numpy as np
import cv2
import pytesseract
from flask import Flask, request
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

app = Flask(__name__)

UPLOAD_PATH = '/home/ubuntu/files'

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
cfg.DATALOADER.NUM_WORKERS = 2
#COCO-Detection/faster_rcnn_R_50_FPN_1x/137257794/model_final_b275ba.pkl
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)


# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.87   # set a custom testing threshold
predictor = DefaultPredictor(cfg)


@app.route('/', methods = ['POST'])
def read_plate():

    if 'plate' not in request.files:
        return 'No se ha seleccionado un archivo válido.', 400
    file = request.files['plate']

    if file.filename == '':
        return 'No se ha seleccionado un archivo válido.', 400

    content = np.asarray(bytearray(file.stream.read()))
    img = cv2.imdecode(content, 0)
    output = predictor(img)

    platePoints = np.array(output["instances"].to("cpu").get("pred_boxes").tensor)

    topx = int(platePoints[0][0])
    topy = int(platePoints[0][1])
    bottomx = int(platePoints[0][2])
    bottomy = int(platePoints[0][3])
    cropped = img[topy:bottomy+1, topx:bottomx+1]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, cropped = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)

    text = pytesseract.image_to_string(cropped, config='--psm 8')

    return text


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)