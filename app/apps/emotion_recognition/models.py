from django.db import models
from PIL import Image


# Create your models here.
class UserImageRecognition(models.Model):
    uploaded_image = models.ImageField(null=False, blank=False)
    final_image = models.ImageField(null=True, blank=True)
    recognized_emotion = models.CharField(max_length=20,null=True, blank=True)
    predicted_emotions = models.CharField(max_length=155,null=True,blank=True)
    created_at = models.DateTimeField(
        auto_now_add=True, null=False, blank=False)

    STATUS_CHOICES = (
        ('PEN', 'Pending'),
        ('COM', 'Complete'),
        ('ERR', 'Error'),
    )
    status = models.CharField(
        max_length=3, choices=STATUS_CHOICES, null=False, blank=False, default='PEN')

    class Meta:
        ordering = ['-created_at']

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        uploaded_img = Image.open(self.uploaded_image.path)
        if uploaded_img.height > 400 or uploaded_img.width > 400:
            output_size = (400, 400)
            uploaded_img.thumbnail(output_size)
            uploaded_img.save(self.uploaded_image.path)
            
    def __str__(self):
        return f'{self.user} - {self.uploaded_image}'
