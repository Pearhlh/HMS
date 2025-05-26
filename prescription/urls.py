from django.urls import path
from . import views

urlpatterns = [
    path('', views.prescription_list, name='prescription_list'),
    path('<int:pk>/', views.prescription_detail, name='prescription_detail'),
    path('create/', views.create_prescription, name='create_prescription'),
    path('<int:pk>/update/', views.update_prescription, name='update_prescription'),
    path('<int:pk>/verify/', views.verify_prescription, name='verify_prescription'),
    path('<int:pk>/dispense/', views.dispense_prescription, name='dispense_prescription'),
    path('medications/', views.medication_list, name='medication_list'),
    path('medications/add/', views.add_medication, name='add_medication'),
]
