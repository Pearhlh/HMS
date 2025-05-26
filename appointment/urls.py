from django.urls import path
from . import views

urlpatterns = [
    path('', views.appointment_list, name='appointment_list'),
    path('book/', views.book_appointment, name='book_appointment'),
    path('<int:pk>/', views.appointment_detail, name='appointment_detail'),
    path('<int:pk>/update/', views.update_appointment, name='update_appointment'),
    path('<int:pk>/cancel/', views.cancel_appointment, name='cancel_appointment'),
    path('<int:pk>/confirm/', views.confirm_appointment, name='confirm_appointment'),
    path('calendar/', views.appointment_calendar, name='appointment_calendar'),
]
