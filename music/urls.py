from django.urls import path
from . import views

app_name = 'Music'

urlpatterns = [
    path('analyze/', views.analyze, name='analyze'),
    path('search/', views.search, name='search')
]
