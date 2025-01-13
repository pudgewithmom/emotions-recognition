from django.urls import path
from . import views


app_name = 'recognition'
urlpatterns = [
    path('recognition/', views.index, name="index"),
    path('api/emotion-recognition/', views.ImageProcessingView.as_view(), name='emotion_recognition'),
    path('recognition/real-time', views.real_time_recognition, name="real_time"),
    path('real-time-stream', views.real_time_stream, name="real_time_video_stream"),
    path('recognition/edit/<pk>', views.RecognitionUpdateView.as_view(), name="recognition_edit"),
    path('recognition/delete/<pk>', views.RecognitionDeleteView.as_view(), name="recognition_delete"),
]
