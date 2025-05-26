from django.urls import path
from . import views

urlpatterns = [
    path('settings/', views.notification_settings, name='notification_settings'),
    path('history/', views.notification_history, name='notification_history'),
    path('test/', views.test_notification, name='test_notification'),
]
