
from django.urls import include, path
from music.views import home, index
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('', home, name='home'),
    path('user/', include('user.urls')),
    path('music/', include('music.urls')),
    path('melon/', include('melon.urls')),
    path('index/', index),
]

# MEDIA 경로 추가
urlpatterns += static(
    settings.MEDIA_URL,
    document_root = settings.MEDIA_ROOT
)