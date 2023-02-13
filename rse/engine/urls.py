from django.contrib import admin
from django.urls import path
from engine import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('search/', views.SearchImage.as_view()),
    path('extract_img/<str:name>', views.show_image_url),
]+ static(settings.STATIC_URL,document_root=settings.STATIC_ROOT)
