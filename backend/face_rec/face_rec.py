import cv2
import pickle
import os
import numpy as np
import face_recognition
import requests
import json
from utils import utils
from face_rec import face_align


USERS_DB_URL = 'http://users_db-service:5000/api/'
MODEL_PROTOPATH = 'NN_models/deploy.prototxt'
MODEL_MODELPATH = 'NN_models/res10_300x300_ssd_iter_140000.caffemodel'
KNN_MODELPATH = 'NN_models/trained_knn_model.clf'
FACE_DETECTOR_CONFIDENSE = 0.5
FACE_RECOGNITION_THRESHOLD = 0.5


def prepare_img(img):
    scale = (img.shape[1] / 300, img.shape[0] / 300)
    image = cv2.resize(img, (300, 300))
    return image, scale


def load_known_faces():
    known_persons = {}
    response = requests.post(USERS_DB_URL + 'get_names', data={})
    user_names = json.loads(response.text)['users']
    for user_name in user_names:
        response = requests.post(USERS_DB_URL + 'get_card', data={'name': user_name})
        cardID = json.loads(response.text)['card_id']

        response = requests.post(USERS_DB_URL + 'get_face', data={'name': user_name})
        face = utils.str2jpeg(json.loads(response.text)['face'])
        face_img = cv2.imdecode(face, cv2.IMREAD_COLOR)
        face_img = cv2.resize(face_img, (360, 480))
        _, face_img_jpeg = cv2.imencode('.jpg', face_img)

        known_persons[user_name] = {
            'face_ID': face_img_jpeg,
            'name': user_name,
            'ID': cardID,
        }
    return known_persons


def calc_encodings(img, boxes):
    enc = face_recognition.face_encodings(img, known_face_locations=boxes, model='small')
    return enc


class FaceRecognizer:
    def __init__(self, logger):
        self.detector = cv2.dnn.readNetFromCaffe(MODEL_PROTOPATH, MODEL_MODELPATH)
        self.detections = None
        self.boxes = None
        self.enc = None
        self.closest_distances = None
        self.are_matches = None
        self.result = False
        self.imageBlob = None
        self.persons_data = []
        self.image = None
        self.confidence = 0
        self.logger = logger
        self.logger.info('Models loaded')
        self.knn_clf = None
        self.load_knn()

        self.known_persons = {}
        self.load_users()

        self.logger.info('Loaded NN_models and classifier')
        for pers in self.known_persons:
            self.logger.info("{} - loaded ID".format(self.known_persons[pers]["name"]))
        self.logger.info('Users loaded')

    def load_knn(self):
        response = requests.post(USERS_DB_URL + 'get_model', data={})
        self.knn_clf = utils.str2model(json.loads(response.text)['model'])
        self.logger.info('KNN predictor loaded')

    def load_users(self):
        self.known_persons = load_known_faces()

    def detect_faces(self, image):
        self.result = False
        self.imageBlob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        self.detector.setInput(self.imageBlob)
        self.detections = self.detector.forward()
        if self.detections.shape[2] > 0:
            self.result = True
        return self.result

    def recognize_face(self, img):
        self.result = False
        self.persons_data = []
        self.image, scale = prepare_img(img)
        (h, w) = self.image.shape[:2]
        self.boxes = []
        if self.detect_faces(self.image):
            for i in range(0, self.detections.shape[2]):
                self.confidence = self.detections[0, 0, i, 2]
                if self.confidence > FACE_DETECTOR_CONFIDENSE:
                    box = self.detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    x0 = int(startX + 0 * w / 10)
                    y0 = int(startY + 0.2 * h / 4)
                    x1 = int(endX - 0 * w / 10)
                    y1 = int(endY)
                    if y0 > y1:
                        t = x0
                        q = y0
                        x0 = x1
                        y0 = y1
                        x1 = t
                        y1 = q
                    if x0 > x1:
                        t = x0
                        q = y0
                        x0 = x1
                        y0 = y1
                        x1 = t
                        y1 = q
                    self.boxes.append((y0, x1, y1, x0))
        unknowns_cnt = len(self.boxes)
        self.enc = []

        if self.boxes:
            self.result = True
            self.enc = calc_encodings(self.image[:, :, ::-1], self.boxes)
            self.closest_distances = self.knn_clf.kneighbors(self.enc, n_neighbors=1)
            self.are_matches = [self.closest_distances[0][i][0] <= FACE_RECOGNITION_THRESHOLD
                                for i in range(len(self.boxes))]
            for predicted_user, face_location, found in zip(self.knn_clf.predict(self.enc), self.boxes,
                                                            self.are_matches):
                y0, x1, y1, x0 = face_location
                x0 = int(x0 * scale[0])
                x1 = int(x1 * scale[0])
                y0 = int(y0 * scale[1])
                y1 = int(y1 * scale[1])
                box = (x0, y0, x1, y1)
                if found:
                    unknowns_cnt -= 1
                    person_found = self.known_persons.get(predicted_user)
                    if person_found is not None:
                        self.persons_data.append((person_found["name"],
                                                  person_found["ID"],
                                                  box))
                else:
                    self.persons_data.append(('unknown',
                                              '0',
                                              box))
        return self.result, self.persons_data, unknowns_cnt
