import requests
import json
import cv2
from backend.utils import utils
import time
import imutils
import os


response = requests.post('http://127.0.0.1:5000/api/update', data={})


cnt = 1
for i in range(1, 102):
    old_file_name = 'validation/foto ({}).jpg'.format(i)
    img = cv2.imread(old_file_name)

    img = imutils.resize(img, height=500)
    _, jpg = cv2.imencode('.jpg', img)
    json_img = utils.jpeg2str(jpg.tobytes())

    t0 = time.time()
    response = requests.post('http://127.0.0.1:5000/api/recognize', data={'image': json_img})
    t1 = time.time()
    print('{:.2f}ms'.format((t1-t0)*1000))
    msg = json.loads(response.text)
    boxes = msg['boxes']
    names = msg['names']

    faces = []
    for box in boxes:
        bbox = json.loads(box)
        x1 = bbox['x0']
        y1 = bbox['y0']
        x2 = bbox['x1']
        y2 = bbox['y1']
        face = img[y1:y2, x1:x2]
        faces.append(face)
    file_name = ''
    for k, face in enumerate(faces):
        user_name = names[k]
        bbox = json.loads(boxes[k])
        x1 = bbox['x0']
        y1 = bbox['y0']
        x2 = bbox['x1']
        y2 = bbox['y1']
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        img = cv2.putText(img, user_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite('validation/post/faces/{}/{}.jpg'.format(user_name, cnt), face)
        cnt = cnt + 1

    folder = os.path.split(old_file_name)[0]
    file_name = os.path.split(old_file_name)[1]
    cv2.imwrite(os.path.join(folder, 'post', file_name), img)
