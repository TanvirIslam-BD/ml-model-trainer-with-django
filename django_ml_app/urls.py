# django_ml_app/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('ml/', include('ml_app.urls')),  # Include ml_app URLs at root level
]
