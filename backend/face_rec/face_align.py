import cv2
import numpy as np
import face_recognition
import argparse


def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def cosine_formula(length_line1, length_line2, length_line3):
    cos_a = -(length_line3 ** 2 - length_line2 ** 2 - length_line1 ** 2) / (2 * length_line2 * length_line1)
    return cos_a


def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def is_between(point1, point2, point3, extra_point):
    c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (extra_point[0] - point1[0])
    c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (extra_point[0] - point2[0])
    c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (extra_point[0] - point3[0])
    if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
        return True
    else:
        return False


def align_face(img, boxes):
    faces = []
    for box in boxes:
        y0, x1, y1, x0 = box
        smallimg = img[y0:y1, x0:x1, :]
        if smallimg.size <= 0:
            continue
        faces.append((smallimg, (x0, y0, x1, y1)))
        """
        face = cv2.resize(smallimg, (300, 300))
        land = face_recognition.face_landmarks(face[:, :, ::-1], [[0, 299, 299, 0]])
        left_eye = land[0]['left_eye']
        right_eye = land[0]['right_eye']
        nose = land[0]['nose_bridge']
        nose_tip = nose[np.argmax(np.array(nose), axis=0)[1]]
        left_eye_center = (int(np.mean([p[0] for p in left_eye])), int(np.mean([p[1] for p in left_eye])))
        right_eye_center = (int(np.mean([p[0] for p in right_eye])), int(np.mean([p[1] for p in right_eye])))
        center_of_forehead = ((left_eye_center[0] + right_eye_center[0]) // 2,
                              (left_eye_center[1] + right_eye_center[1]) // 2)
        center_pred = (150, 150)

        length_line1 = distance(center_of_forehead, nose_tip)
        length_line2 = distance(center_pred, nose_tip)
        length_line3 = distance(center_pred, center_of_forehead)

        cos_a = cosine_formula(length_line1, length_line2, length_line3)
        angle = np.arccos(cos_a)

        rotated_point = rotate_point(nose_tip, center_of_forehead, angle)
        rotated_point = (int(rotated_point[0]), int(rotated_point[1]))
        if is_between(nose_tip, center_of_forehead, center_pred, rotated_point):
            angle = np.degrees(-angle)
        else:
            angle = np.degrees(angle)

        M = cv2.getRotationMatrix2D(center_pred, angle, 1.0)
        face = cv2.warpAffine(face, M, (300, 300))
        faces.append((face, (x0, y0, x1, y1)))
        """
    return faces


if __name__ == '__main__':
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, default='E:\\PythonProjects\\face_rec-service\\1.jpg', type=str,
                    help="path to image")
    args = vars(ap.parse_args())
    img = cv2.imread(args['image'])
    boxes = face_recognition.face_locations(img)
    faces = align_face(img, boxes)
    folder, _ = os.path.split(os.path.abspath(args['image']))
    cnt = 0
    for face in faces:
        fname = os.path.join(folder, 'face_{}.jpg'.format(cnt))
        cnt = cnt + 1
        cv2.imwrite(fname, face)
        enc = face_recognition.face_encodings(face[:, :, ::-1], known_face_locations=[[0, 299, 299, 0]], model='small')
        print(enc)
