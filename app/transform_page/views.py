from .models import Videos, VideosLogosTable

from django.shortcuts import render
from .forms import UploadFileForm
from django.views.decorators.csrf import ensure_csrf_cookie
from .object_detection.object_detection_algorithm import ODM
from .object_detection.logo_classification_algorithm import LogoClassifier
import os


@ensure_csrf_cookie
def index(request):
    print(os.getcwd())
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():

            file = request.FILES['file']
            title = file.name
            content = Videos(title=title, file=file)
            content.save()
            # handle_uploaded_file(file)
            model = ODM()
            model.detect_logos(
                # modelul salvat
                path_to_model=os.path.join('transform_page\object_detection', 'fix1_ports_model.pth'),
                # path to video input
                path_to_video=os.path.join(r'media\file', file.name),
                # path where to save result video and csv result
                path_to_save=r'media\file_blurred',
                file_name=file.name,
            )
            classification_model = LogoClassifier()
            output_classification_model = classification_model.classify_logos(
                boxes_path=os.path.join(r'media\file_blurred', 'logo_boxes.csv'),
                model_path=os.path.join('transform_page\object_detection', 'logos_model_fin.pth'),
                video_path=os.path.join(r'media\file', file.name)
            )

            content_logos = VideosLogosTable(title=title, dictionary_of_values=str(output_classification_model))
            content_logos.save()

            return render(request, "transform_page/left-sidebar.html", {'filename': file.name})
    else:

        form = UploadFileForm()
    return render(request, 'transform_page/left-sidebar.html', {'form': form})


def handle_uploaded_file(f):

    with open(os.path.join(r'media\file', f.name), 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
