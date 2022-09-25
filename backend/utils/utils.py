import pickle

import numpy as np
import base64


def str2jpeg(json_string):
    img_jpeg = np.frombuffer(base64.decodebytes(json_string.encode('UTF-8')), dtype=np.uint8)
    img_jpeg = img_jpeg.reshape(len(img_jpeg), 1)
    return img_jpeg


def str2encoding(json_string):
    enc = np.frombuffer(base64.decodebytes(json_string.encode('UTF-8')), dtype=np.float64)
    enc = enc.reshape(len(enc))
    return enc


def jpeg2str(img_jpeg_bytes):
    json_string = base64.encodebytes(img_jpeg_bytes).decode('UTF-8')
    return json_string


def encoding2str(encoding_bytes):
    json_enc = base64.encodebytes(encoding_bytes).decode('UTF-8')
    return json_enc


def model2str(model):
    json_string = base64.encodebytes(pickle.dumps(model)).decode('UTF-8')
    return json_string


def str2model(json_string):
    model = pickle.loads(base64.decodebytes(json_string.encode('UTF-8')))
    return model
