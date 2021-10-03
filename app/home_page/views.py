from django.shortcuts import render
from django.http import HttpResponseRedirect


def index(request):
    if request.POST:
        return render(request, "transform_page/left-sidebar.html")
    return render(request, "home_page/index.html")


def home(response):
    return render(response, "home_page/index.html")
