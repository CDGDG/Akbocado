from django.urls import path
from . import views

app_name = 'Music'

urlpatterns = [
    path('analyze/', views.analyze, name='analyze'),
    path('analyze_lyrics/',views.analyze_lyrics,name = 'analyze_lyrics'),
    path('search/', views.search, name='search'),
    path('checkImage/',views.checkAkboImage, name='checkImage'),
]
