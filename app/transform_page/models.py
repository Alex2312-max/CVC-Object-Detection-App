from django.db import models

# Create your models here.


class Videos(models.Model):
    title = models.CharField(max_length=100)
    file = models.FileField(default=None, upload_to='file')

    class Meta:
        verbose_name = 'file'
        verbose_name_plural = 'files'

    def __str__(self):
        return self.title


class VideosLogosTable(models.Model):
    title = models.CharField(max_length=100)
    dictionary_of_values = models.CharField(max_length=400)

    def __str__(self):
        return self.title

