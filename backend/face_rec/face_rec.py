import requests
import json
from utils import utils
import face_recognition
import cv2
import numpy as np
import imutils
import pickle


USERS_DB_URL = 'http://users_db-service:5000/api/'
MODEL_PROTOPATH = 'NN_models/deploy.prototxt'
MODEL_MODELPATH = 'NN_models/res10_300x300_ssd_iter_140000.caffemodel'
KNN_MODELPATH = 'NN_models/trained_knn_model.clf'
CONFIDENCE_THRESHOLD = 0.7
MINIMAL_FACE_AREA = 20 * 20
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


def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def area(pt1, pt2):
    a = distance(pt1, (pt2[0], pt1[1]))
    b = distance(pt1, (pt1[0], pt2[1]))
    return a * b


def rotated_rect_with_max_area(w, h, angle):
    """
    https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(np.sin(angle)), abs(np.cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr, hr = (x/sin_a, x/cos_a) if width_is_longer else (x/cos_a, x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr, hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

    return int(wr), int(hr)


def calc_angle(land_right, land_left):
    left_eye = land_left
    right_eye = land_right
    left_eye_center = (int(np.mean([p[0] for p in left_eye])), int(np.mean([p[1] for p in left_eye])))
    right_eye_center = (int(np.mean([p[0] for p in right_eye])), int(np.mean([p[1] for p in right_eye])))

    if left_eye_center[1] > right_eye_center[1]:
        some_point = (right_eye_center[0], left_eye_center[1])
        direction = -1
    else:
        some_point = (left_eye_center[0], right_eye_center[1])
        direction = 1

    line_a = distance(left_eye_center, some_point)
    line_b = distance(right_eye_center, some_point)
    if direction < 0:
        angle = np.arctan2(line_b, line_a)
    else:
        angle = np.arctan2(line_a, line_b)

    angle = np.degrees(angle * direction)
    return angle


def align_face(face):
    orig_size = face.shape[0:2][::-1]
    face_small = cv2.resize(face, (300, 300))
    land = face_recognition.face_landmarks(face_small[:, :, ::-1], [[0, 299, 299, 0]])

    angle = calc_angle(land[0]['right_eye'], land[0]['left_eye'])

    face = imutils.rotate_bound(face, -angle)
    w, h = rotated_rect_with_max_area(orig_size[0], orig_size[1], np.radians(angle))
    h0, w0, _ = face.shape
    face = face[int((h0 - h)/2):int((h0 + h)/2), int((w0-w)/2):int((w0+w)/2)]
    face = cv2.resize(face, orig_size)
    """
    m = cv2.getRotationMatrix2D((int(orig_size[0]/2), int(orig_size[1]/2)), angle, 1.0)
    # face = cv2.circle(face_small, center=left_eye_center, color=(0, 0, 255), radius=2, thickness=2)
    # face = cv2.circle(face, center=right_eye_center, color=(0, 0, 255), radius=2, thickness=2)
    # face = cv2.circle(face, center=some_point, color=(0, 0, 255), radius=2, thickness=2)
    
    # face = cv2.warpAffine(face, m, face.shape[0:2][::-1])
    print(orig_size, w, h)
    face = cv2.warpAffine(face, m, (int(orig_size[0]*1.5), int(orig_size[1]*1.5)))
    face = cv2.resize(face, orig_size)"""
    return face


def calc_encoding(face):
    h, w, _ = face.shape
    enc = face_recognition.face_encodings(face[:, :, ::-1], known_face_locations=[(0, w, h, 0)], model='small')
    return enc[0]


def calc_encodings(img, boxes):
    enc = face_recognition.face_encodings(img, known_face_locations=boxes, model='small')
    return enc


class FaceRecognizer:
    def __init__(self, logger):
        self.detector = cv2.dnn.readNetFromCaffe(MODEL_PROTOPATH, MODEL_MODELPATH)
        self.detector.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        self.logger = logger
        self.logger.info('Models loaded')
        self.classifier = None
        self.load_knn()
        self.known_persons = {}
        self.load_users()

        self.logger.info('Loaded NN_models and classifier')
        for pers in self.known_persons:
            self.logger.info("{} - loaded ID".format(self.known_persons[pers]["name"]))
        self.logger.info('Users loaded')

    def load_knn(self):
        try:
            response = requests.post(USERS_DB_URL + 'get_model', data={})
            self.classifier = utils.str2model(json.loads(response.text)['model'])
            self.logger.info('KNN predictor loaded')
        except Exception:
            self.classifier = None
            self.logger.info('KNN predictor empty')

    def load_users(self):
        self.known_persons = load_known_faces()

    def detect_face(self, image):
        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1.0, size=(300, 300), mean=[104, 117, 123], swapRB=False, crop=False,
        )
        self.detector.setInput(blob)
        detections = self.detector.forward()
        bounding_boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CONFIDENCE_THRESHOLD:
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                if x1 < width and x2 < width and y1 < height and y2 < height:
                    if area((x1, y1), (x2, y2)) > MINIMAL_FACE_AREA:
                        bounding_boxes.append([x1, y1, x2, y2])
        faces = []
        new_boxes = []
        for box in bounding_boxes:
            x1, y1, x2, y2 = box
            face = image[y1:y2, x1:x2]
            if face.size > 0:
                aligned_face = align_face(face)
                faces.append(aligned_face)
                new_boxes.append(box)
        return faces, new_boxes

    def recognize_face(self, image):
        faces, boxes = self.detect_face(image)
        names = []
        persons_data = []
        result = False
        unknowns_cnt = 0
        for k, face in enumerate(faces):
            result = True
            face = imutils.resize(face, height=112)
            enc = calc_encoding(face)
            if self.classifier is None:
                user_name = 'unknown'
                unknowns_cnt = unknowns_cnt + 1
                persons_data.append(('unknown',
                                     '0',
                                     boxes[k]))
            else:
                closest = self.classifier.kneighbors([enc], n_neighbors=1)
                if closest[0][0][0] < FACE_RECOGNITION_THRESHOLD:
                    user_name = self.classifier.predict([enc])[0]
                    persons_data.append((user_name,
                                         self.known_persons[user_name]["ID"],
                                         boxes[k]))
                else:
                    user_name = 'unknown'
                    unknowns_cnt = unknowns_cnt + 1
                    persons_data.append(('unknown',
                                         '0',
                                         boxes[k]))
            names.append(user_name)
        return result, persons_data, unknowns_cnt
