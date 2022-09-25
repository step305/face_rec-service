import requests
import json
import cv2
from backend.utils import utils
import time


response = requests.post('http://127.0.0.1:5000/api/update', data={})


for name in range(1, 7):
    for i in range(1, 7):
        img = cv2.imread('test_train/user{}/{}.jpg'.format(name, i))

        img = cv2.resize(img, (800, 600))
        _, jpg = cv2.imencode('.jpg', img)
        json_img = utils.jpeg2str(jpg.tobytes())

        t0 = time.time()
        response = requests.post('http://127.0.0.1:5000/api/recognize', data={'image': json_img})
        t1 = time.time()
        print('{:.2f}ms'.format((t1-t0)*1000))
        msg = json.loads(response.text)
        boxes = msg['boxes']
        box = json.loads(boxes[0])
        print(msg['unknown'], msg['names'])

        face = img[box['y0']:box['y1'], box['x0']:box['x1']]
        while cv2.waitKey(1) != 27:
            cv2.imshow('a', face)
