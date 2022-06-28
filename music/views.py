from django.shortcuts import render
from matplotlib import artist
from music.models import Input

def home(request):
    return render(request, 'home.html')

def index(request):
    return render(request, 'index.html')

def analyze(request):

    akbo = None; title = None; artist = None

    akbo = request.FILES.get('akbo')

    image = Input(
        img=akbo,
        img_original=akbo.name,
    )
    image.save()

    # 분석 작업
    context = {
        'image': image,
        'title': title,
        'artist': artist,
    }

    return render(request, 'result.html', context)