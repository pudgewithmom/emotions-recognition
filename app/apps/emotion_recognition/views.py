from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods
from django.http import HttpResponse, StreamingHttpResponse
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import UpdateView, DeleteView
from django.contrib import messages
from django.urls import reverse_lazy

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from .serializers import ImageSerializer
from PIL import Image

from .models import UserImageRecognition
from .emotion_recognition import  emotionVideo
from .tasks import proccess_uploaded_image, process_image_from_api, convert_image_to_bytes
from .forms import RecognitionEditForm



@require_http_methods(['GET', 'POST'])
def index(request):
    try:

        # GET method, return HTML page
        if request.method == 'GET':
            samples = UserImageRecognition.objects.all()
            return render(request, 'recognition/index.html', {'samples': samples, })

        if request.FILES and request.method == 'POST':
            for f in request.FILES.getlist('uploaded_file'):
                uploaded_image = f
                image_data = UserImageRecognition.objects.create(uploaded_image=uploaded_image)

                proccess_uploaded_image(image_data.id)

        return redirect('recognition:index')

    except Exception as e:

        image_data.status = 'ERR'
        image_data.error_occurred = True
        image_data.error_message = str(e)
        image_data.save()

        return HttpResponse(f'Error: {str(e)}')
    


@require_http_methods(['GET', 'POST'])
def real_time_recognition(request):
    return render(request, 'recognition/real_time.html')

def real_time_stream(request):
    return StreamingHttpResponse(emotionVideo(),content_type="multipart/x-mixed-replace;boundary=frame")


class RecognitionUpdateView(LoginRequiredMixin, UpdateView):
    model = UserImageRecognition
    form_class = RecognitionEditForm
    template_name = "recognition/recognition_edit.html"

    def get(self, request, pk):
        self.object = self.get_object()
        
        context = self.get_context_data(object=self.object)
        return self.render_to_response(context)
    
    def get_success_url(self, **kwargs):
        pk = self.object.pk
        messages.success(self.request, 'Запись была успешно изменена!')         
        return reverse_lazy('recognition:recognition_edit', args=(pk,))


class RecognitionDeleteView(LoginRequiredMixin, DeleteView):
    model = UserImageRecognition
    template_name = "recognition/recognition_delete.html"

    def delete(self, request, pk):
        return super().delete(request, pk)

    def get_success_url(self, **kwargs):
        obj = self.get_object()
        messages.success(self.request, 'Запись  была успешно удалёна!')         
        return reverse_lazy('recognition:index')
    
    
class ImageProcessingView(APIView):
    parser_classes = (MultiPartParser, FormParser)  # Для обработки загруженных файлов
    serializer_class = ImageSerializer

    def post(self, request, format=None):
        """
        Обрабатывает изображение, переданное через API, и возвращает финальное изображение с эмоциями.
        """
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            image_file = serializer.validated_data['image']
            
            try:
                # Обработка изображения
                final_image, predicted_emotions, recognized_emotion = process_image_from_api(image_file)
                output_image = convert_image_to_bytes(final_image)

                # Формируем ответ
                response_data = {
                    "predicted_emotions": predicted_emotions,
                    "recognized_emotion": recognized_emotion,
                }
                response = Response(response_data, status=status.HTTP_200_OK)
                response['Content-Type'] = 'image/jpeg'
                response.content = output_image.getvalue()
                return response
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)