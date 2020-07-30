from flask import Flask, jsonify, request
import logging
handler = logging.FileHandler('./app.log')  # errors logged to this file
handler.setLevel(logging.ERROR)  # only log errors and above
import face_recognition
import torch
import urllib
import models
import cv2
import numpy as np
import imutils
import time
import ctypes
import os

print("[File]: ",torch.__file__)
print("[Version]: ",torch.__version__)

protoPath = "./face_detector/deploy.prototxt"
modelPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

protoPath2 = "./face_alignment/2_deploy.prototxt"
modelPath2 = "./face_alignment/2_solver_iter_800000.caffemodel"
net2 = cv2.dnn.readNetFromCaffe(protoPath2, modelPath2)

print("cv2 Models Loaded")

model_name = "MyresNet18"
load_model_path = "a8.pth"
model = models.MyresNet18()
model.load_state_dict(torch.load(load_model_path, map_location=torch.device('cpu')))
# model = torch.load(load_model_path,map_location=torch.device('cpu'))
model.train(False)

print("Model Loaded")



ATTACK = 1
GENUINE = 0
thresh = 0.7
server = Flask(__name__)
server.logger.addHandler(handler)

def detector(img):
    frame = imutils.resize(img, width=600)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (400, 400))
            return face

def crop_with_ldmk(image, landmark):
    scale = 3.5
    image_size = 224
    ct_x, std_x = landmark[:, 0].mean(), landmark[:, 0].std()
    ct_y, std_y = landmark[:, 1].mean(), landmark[:, 1].std()

    std_x, std_y = scale * std_x, scale * std_y

    src = np.float32([(ct_x, ct_y), (ct_x + std_x, ct_y + std_y), (ct_x + std_x, ct_y)])
    dst = np.float32([((image_size - 1) / 2.0, (image_size - 1) / 2.0),
                      ((image_size - 1), (image_size - 1)),
                      ((image_size - 1), (image_size - 1) / 2.0)])
    retval = cv2.getAffineTransform(src, dst)
    result = cv2.warpAffine(image, retval, (image_size, image_size), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
    return result

def demo(img):
    print("Demo called")
    data= np.transpose(np.array(img, dtype=np.float32), (2, 0, 1))
    data = data[np.newaxis, :]
    data = torch.FloatTensor(data)
    with torch.no_grad():
        outputs = model(data)
        outputs = torch.softmax(outputs, dim=-1)
        preds = outputs.to('cpu').numpy()
        attack_prob = preds[:, ATTACK]
    return  attack_prob


@server.route("/", methods = ["POST"])
def func():
    data = request.json
    urls = list(data["urls"])
    unknown_url = data["unknown_url"]
    
    faces = 0

    try:
    	resp = urllib.request.urlopen(unknown_url)
    	image = np.asarray(bytearray(resp.read()), dtype="uint8")
    	frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
    	image = frame
    except:
        return jsonify(statusCode = 200, body= "Error",error= 5,message= "Unknown Image is not accessible",exception = "A URL not accessible")

    frame = imutils.resize(frame, width=600)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)

    detections = net.forward()

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            faces = faces+1
            if(faces>1):
                break

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            sx = startX
            sy = startY
            ex = endX
            ey = endY

            ww = (endX - startX) // 10
            hh = (endY - startY) // 5

            startX = startX - ww
            startY = startY + hh
            endX = endX + ww
            endY = endY + hh

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            x1 = int(startX)
            y1 = int(startY)
            x2 = int(endX)
            y2 = int(endY)

            roi = frame[y1:y2, x1:x2]
            gary_frame = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            resize_mat = np.float32(gary_frame)
            m = np.zeros((40, 40))
            sd = np.zeros((40, 40))
            mean, std_dev = cv2.meanStdDev(resize_mat, m, sd)
            new_m = mean[0][0]
            new_sd = std_dev[0][0]
            new_frame = (resize_mat - new_m) / (0.000001 + new_sd)
            blob2 = cv2.dnn.blobFromImage(cv2.resize(new_frame, (40, 40)), 1.0, (40, 40), (0, 0, 0))
            net2.setInput(blob2)
            align = net2.forward()

            aligns = []
            alignss = []
            for i in range(0, 68):
                align1 = []
                x = align[0][2 * i] * (x2 - x1) + x1
                y = align[0][2 * i + 1] * (y2 - y1) + y1
                cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), 2)
                align1.append(int(x))
                align1.append(int(y))
                aligns.append(align1)

            alignss.append(aligns)
            ldmk = np.asarray(alignss, dtype=np.float32)
            ldmk = ldmk[np.argsort(np.std(ldmk[:,:,1],axis=1))[-1]]
            img = crop_with_ldmk(frame,ldmk)

            attack_prob = demo(img)

            true_prob = 1 - attack_prob
            if attack_prob > thresh:
                label = 'fake'
                break
            else:
                label = 'true'

    error = -1
    body = "False"
    message = ""
    exception = ""

    if(faces==0):
        error = 1
        body = "Error"
        message = "No Face Detected"
    elif(faces>1):
        error = 2
        body = "Error"
        message = "Multiple Faces Detected"
    elif(faces==1 and label == 'fake'):
        error = 3
        body = "Error"
        message = "Only 1 Face Detected but is fake"

    elif(faces==1 and label == 'true'):

        boxes = face_recognition.face_locations(image)

        if(len(boxes)==0):
            message = "No Face Detected"
            error = 1
            body = "Error"

            # return False, error, msg

        else:
            unknown = face_recognition.face_encodings(image,boxes)[0]

            for im_path in urls:
                try:
                    resp = urllib.request.urlopen(im_path)
                    image = np.asarray(bytearray(resp.read()), dtype="uint8")
                except:
                    exception = "A URL not accessible"
                    continue

                image = cv2.imdecode(image, cv2.IMREAD_COLOR)

                boxes = face_recognition.face_locations(image)
                if(len(boxes)==0):
                    continue

                known = face_recognition.face_encodings(image,boxes)[0]
                matches = face_recognition.compare_faces([unknown], known)
                if(matches[0]==True):
                    error = -1
                    message = "Validated"
                    body = "True"
                    break

            if(body!="True"):
                error = 4
                message = "Face not Validated"
                body = "Error"

    return jsonify(statusCode = 200, body= body,error= error,message= message,exception = exception)
