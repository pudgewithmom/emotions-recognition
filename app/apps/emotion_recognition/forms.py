from django import forms

from .models import UserImageRecognition


class RecognitionEditForm(forms.ModelForm):
    class Meta:
        model = UserImageRecognition
        fields = ('__all__')