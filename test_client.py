from __future__ import print_function
import requests
import json
import cv2
from backend.utils import utils
import time
import argparse
import pickle
import codecs


response = requests.post('http://127.0.0.1:5000/api/downloadID', data={})
print(json.loads(response.text))

img = cv2.imread('test.jpg')
_, jpg = cv2.imencode('.jpg', img)
json_img = utils.jpeg2str(jpg.tobytes())

response = requests.post('http://127.0.0.1:5000/api/recognize', data={'image': json_img})
print(json.loads(response.text))
msg = json.loads(response.text)
boxes = msg['boxes']

cnt = 0

for box in boxes:
    box = json.loads(box)

    face = img[box['y0']:box['y1'], box['x0']:box['x1']]
    while cv2.waitKey(1) != 27:
        cv2.imshow('a', face)

    _, jpg = cv2.imencode('.jpg', face)
    json_img = utils.jpeg2str(jpg.tobytes())

    response = requests.post('http://127.0.0.1:5000/api/get_encoding', data={'image': json_img})
    enc = utils.str2encoding(json.loads(response.text)['enc'])

    json_enc = utils.encoding2str(enc.tobytes())

    response = requests.post('http://127.0.0.1:8000/api/add_photo', data={'image': json_img,
                                                                          'name': 'user{}'.format(cnt),
                                                                          'encoding': json_enc})
    print(json.loads(response.text))

    cnt = cnt + 1
    print(json.loads(response.text))
    response = requests.post('http://127.0.0.1:8000/api/train_model', data={})
    print(json.loads(response.text))






