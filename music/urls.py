from django.urls import path
from . import views

app_name = 'Music'

urlpatterns = [
    path('analyze/', views.analyze, name='analyze'),
    path('analyze_type/<type>/',views.analyze_type,name = 'analyze_type'),
    path('search/', views.search, name='search'),
    path('checkImage/',views.checkAkboImage, name='checkImage'),
]
