import requests
import json
import cv2
from backend.utils import utils

for name in range(1, 7):
    flag = 0
    for i in range(1, 7):
        img = cv2.imread('test_train/user{}/{}.jpg'.format(name, i))

        _, jpg = cv2.imencode('.jpg', img)
        json_img = utils.jpeg2str(jpg.tobytes())

        response = requests.post('http://127.0.0.1:5000/api/recognize', data={'image': json_img})
        msg = json.loads(response.text)
        boxes = msg['boxes']
        if len(boxes) > 0:
            flag = flag + 1
            box = json.loads(boxes[0])
            print(box)

            face = img[box['y0']:box['y1'], box['x0']:box['x1']]
            face = cv2.resize(face, (300, 300))
            """
            while cv2.waitKey(1) != 27:
                cv2.imshow('a', face)
            """
            _, jpg = cv2.imencode('.jpg', face)
            json_img = utils.jpeg2str(jpg.tobytes())

            if flag == 1:
                response = requests.post('http://127.0.0.1:8000/api/add_new', data={'image': json_img,
                                                                                    'name': 'user{}'.format(name),
                                                                                    'card_id': '00000'})
                print(json.loads(response.text))

            response = requests.post('http://127.0.0.1:5000/api/get_encoding', data={'image': json_img})
            enc = utils.str2encoding(json.loads(response.text)['enc'])

            json_enc = utils.encoding2str(enc.tobytes())

            response = requests.post('http://127.0.0.1:8000/api/add_photo', data={'image': json_img,
                                                                                  'name': 'user{}'.format(name),
                                                                                  'encoding': json_enc})
            print(json.loads(response.text))

response = requests.post('http://127.0.0.1:8000/api/train_model', data={})
print(json.loads(response.text))

