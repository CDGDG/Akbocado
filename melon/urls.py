from django.urls import path
from . import views

app_name = 'Melon'

urlpatterns = [
    path('analyze/', views.analyze, name='analyze'),
]
