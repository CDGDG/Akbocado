from django.urls import path
from . import views

app_name = 'User'

urlpatterns = [
    path('login/', views.login, name='login'),
    path('logout/', views.logout, name='logout'),
    path('join/', views.join, name='join'),
    path('checkid/', views.checkid, name='checkid'),
    path('myakbo/', views.myakbo, name='myakbo'),
    path('save/', views.save, name='save'),
    path('akboinfo/<int:pk>/', views.akboinfo, name='akboinfo')
]
