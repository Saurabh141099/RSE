from django.contrib import admin
from django.urls import path, include
from engine.views import apiOverview

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',apiOverview),
    path('engine/', include('engine.urls')),
]
