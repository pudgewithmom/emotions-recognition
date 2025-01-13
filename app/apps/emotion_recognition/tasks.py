from django.shortcuts import get_object_or_404
from django.core.files.base import ContentFile

from .models import UserImageRecognition
from .emotion_recognition import emotionImage, emotionImageFromArray

from io import BytesIO
import cv2
import numpy as np
from PIL import Image


def proccess_uploaded_image(image_data_id):
    image_data = None
    image_data = get_object_or_404(UserImageRecognition, pk=image_data_id)

    final_image, predicted_emotions, recognized_emotion = emotionImage(
        image_data.uploaded_image.name)
    final_image = converter_to_django_file(final_image)
    
    image_data.final_image = final_image
    image_data.predicted_emotions = predicted_emotions
    image_data.recognized_emotion = recognized_emotion
    image_data.status = "COM"
    image_data.save()

def process_image_from_api(image_file):
    """
    Обрабатывает изображение, переданное через API, и возвращает финальное изображение и данные эмоций.

    Args:
        image_file (InMemoryUploadedFile): Исходное изображение, полученное через API.

    Returns:
        tuple: (final_image, predicted_emotions, recognized_emotion)
            - final_image: обработанное изображение в формате numpy array.
            - predicted_emotions: словарь с предсказанными эмоциями и их вероятностями.
            - recognized_emotion: самая вероятная эмоция.
    """
    # Конвертируем загруженный файл в OpenCV-совместимый формат
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Обрабатываем изображение с помощью emotionImageFromArray
    final_image, predicted_emotions, recognized_emotion = emotionImageFromArray(image)

    return final_image, predicted_emotions, recognized_emotion

def converter_to_django_file(image):
    img_io = BytesIO()
    image = Image.fromarray(image)
    image.save(img_io, format='JPEG',  quality=100)
    img_content = ContentFile(img_io.getvalue(), 'final_image.jpg')
    
    return img_content

def convert_image_to_bytes(image):
    # Конвертируем обработанное изображение в байты
    _, buffer = cv2.imencode('.jpg', image)
    output_image = BytesIO(buffer)
    
    return output_image