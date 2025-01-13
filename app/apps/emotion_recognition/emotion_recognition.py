
from tensorflow import keras
from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras.utils import img_to_array
import keras.utils as image 

import cv2
import numpy as np
import os

from django_app.settings import BASE_DIR


model = Sequential()

model = model_from_json(open(
    os.path.join(BASE_DIR,'model/model_4layer_2_2_pool.json'), "r").read())

model.load_weights(os.path.join(
			BASE_DIR,'model/model_4layer_2_2_pool.h5'))

class_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
                3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
classes = list(class_labels.values())

face_classifier = cv2.CascadeClassifier(os.path.join(
			BASE_DIR,'model/haarcascade_frontalface.xml'))

camera = cv2.VideoCapture(0)


def text_on_detected_boxes(text, text_x, text_y, image, font_scale=1,
                           font=cv2.FONT_HERSHEY_SIMPLEX,
                           FONT_COLOR=(0, 0, 0),
                           FONT_THICKNESS=2,
                           rectangle_bgr=(0, 255, 0)):
    (text_width, text_height) = cv2.getTextSize(
        text, font, fontScale=font_scale, thickness=2)[0]
    box_coords = ((text_x-10, text_y+4), (text_x +
                  text_width+10, text_y - text_height-5))
    cv2.rectangle(image, box_coords[0],
                  box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(image, text, (text_x, text_y), font,
                fontScale=font_scale, color=FONT_COLOR, thickness=FONT_THICKNESS)


def face_detector_image(img):
    """
    Обнаружение лиц на изображении.

    Args:
        img (numpy array): Исходное изображение.

    Returns:
        tuple: (rects, allfaces, img) - координаты лиц, обрезанные лица и изображение с рамками.
    """
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces == ():
        return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), img
    allfaces = []
    rects = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        allfaces.append(roi_gray)
        rects.append((x, w, y, h))
    return rects, allfaces, img 


def emotionImage(imgPath):
    img = cv2.imread(BASE_DIR + '\\media\\' + imgPath)
    rects, faces, image = face_detector_image(img)
    i = 0
    for face in faces:
        roi = face.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = model.predict(roi)[0]
        label = class_labels[preds.argmax()]
        label_position = (
            rects[i][0] + int((rects[i][1] / 2)), abs(rects[i][2] - 10))
        i += 1

        # Отрисовка текста и рамок
        text_on_detected_boxes(
            label, label_position[0], label_position[1], image)

        precentages = dict(zip(classes, preds*100))

    return image, precentages, label


def emotionImageFromArray(img_array):
    """
    Обрабатывает изображение и возвращает результат обработки.

    Args:
        img_array (numpy array): Исходное изображение (numpy array).

    Returns:
        tuple: (image, precentages, label)
            - image: Изображение с рамками и текстом эмоций.
            - precentages: Вероятности каждой эмоции.
            - label: Определенная эмоция.
    """
    rects, faces, image = face_detector_image(img_array)
    i = 0
    for face in faces:
        roi = face.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = model.predict(roi)[0]
        label = class_labels[preds.argmax()]
        label_position = (
            rects[i][0] + int((rects[i][1] / 2)), abs(rects[i][2] - 10))
        i += 1

        # Отрисовка текста и рамок
        text_on_detected_boxes(
            label, label_position[0], label_position[1], image)

        precentages = dict(zip(classes, preds*100))

    return image, precentages, label

def face_detector_video(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), img
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        roi_gray = gray[y:y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    return (x, w, y, h), roi_gray, img


def emotionVideo():
    while True:
        ret, frame = camera.read()
        rect, face, image = face_detector_video(frame)
        if np.sum([face]) != 0.0:
            roi = face.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = model.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (rect[0] + rect[1]//50, rect[2] + rect[3]//50)
            text_on_detected_boxes(label, label_position[0], label_position[1], image) 
            fps = camera.get(cv2.CAP_PROP_FPS)
            cv2.putText(image, str(fps),(5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(image, "No Face Found", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', image)
        
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  



def gen_frames():                                     
    while True:
        success, frame = camera.read()
        if not success:
            cv2.putText(image, "No Face Found", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            break
        else:
            gray_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        
            faces_detected = face_classifier.detectMultiScale(gray_img, 1.32, 5)  
            
        
            for (x,y,w,h) in faces_detected:
                
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=7)  
                roi_gray=gray_img[y:y+w,x:x+h]          
                roi_gray=cv2.resize(roi_gray,(48,48))  
                img_pixels = image.img_to_array(roi_gray)  
                img_pixels = np.expand_dims(img_pixels, axis = 0)  
                img_pixels /= 255  
        
                predictions = model.predict(img_pixels)  
        
                max_index = np.argmax(predictions[0])   
        
                emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']  
                predicted_emotion = emotions[max_index]  
                
                cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)  
        
            resized_img = cv2.resize(frame, (600, 400))  
            
            ret, buffer = cv2.imencode('.jpg', frame)
            
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  