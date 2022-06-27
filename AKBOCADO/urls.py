
from django.urls import include, path
from music.views import home, index

urlpatterns = [
    path('', home, name='home'),
    path('user/', include('user.urls')),
    path('index/', index),
]
