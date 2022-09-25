import json
import requests
from flask import Flask, request, Response, render_template, jsonify
import jsonpickle
import base64
from face_rec import face_rec
import argparse
import logging
from utils import utils
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--port", required=True, default=5000, type=int,
                help="path to input dataset")
args = vars(ap.parse_args())

logger = logging.getLogger('recognition_thread_at_port_' + str(args['port']))
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('Logs/log_recognition_thread_at_port_' + str(args['port']) + '.txt')
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

recognizer = face_rec.FaceRecognizer(logger)

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/api/downloadID', methods=['POST'])
def download_request():
    users = recognizer.known_persons
    names = []
    card_ids = []
    faces = []
    for user in users:
        names.append(users[user]['name'])
        card_ids.append(users[user]['ID'])
        json_img = utils.jpeg2str(users[user]['face_ID'].tobytes())
        faces.append(json_img)

    msg = {'names': names, 'IDs': card_ids, 'faces': faces}
    return jsonify(msg)


@app.route('/api/update', methods=['POST'])
def update_request():
    recognizer.load_users()
    recognizer.load_knn()
    return jsonify({'response': 'ok'})


@app.route('/api/recognize', methods=['POST'])
def recognize_request():
    r = request
    img_jpg = utils.str2jpeg(r.form['image'])
    img = cv2.imdecode(img_jpg, cv2.IMREAD_COLOR)

    (res, persons, unknowns_cnt) = recognizer.recognize_face(img)
    names = []
    card_ids = []
    boxes = []
    if res:
        for person in persons:
            name, card_id, box = person
            names.append(name)
            card_ids.append(card_id)
            x0, y0, x1, y1 = box
            s = '{' + "'x0': {0}, 'y0': {1}, 'x1': {2}, 'y1': {3}".format(x0, y0, x1, y1) + '}'
            boxes.append(s.replace("'", "\""))

    msg = {'names': names, 'card_ids': card_ids, 'boxes': boxes, 'unknown': str(unknowns_cnt)}
    return jsonify(msg)


@app.route('/api/get_encoding', methods=['POST'])
def get_encoding_request():
    r = request
    img_jpg = utils.str2jpeg(r.form['image'])
    img = cv2.imdecode(img_jpg, cv2.IMREAD_COLOR)

    img, _ = face_rec.prepare_img(img)
    enc = face_rec.calc_encodings(img, [(0, 299, 299, 0)])[0]
    json_enc = utils.encoding2str(enc.tobytes())

    msg = {'enc': json_enc}
    return jsonify(msg)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
