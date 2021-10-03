from .models import Videos
from django import forms


class UploadFileForm(forms.Form):

    file = forms.FileField()

    class Meta:
        abstract = True
        model = Videos
        fields = ["file"]
