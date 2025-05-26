from django.urls import path
from . import views

app_name = 'doctor'

urlpatterns = [
    # Dashboard routes
    path('', views.dashboard, name='dashboard'),
    
    # Appointment routes
    path('appointments/', views.appointment_list, name='appointments'),
    path('appointment/<int:pk>/manage/', views.appointment_manage, name='appointment_manage'),
    
    # Patient routes
    path('patients/', views.patient_list, name='patients'),
    path('patient/<int:pk>/', views.patient_detail, name='patient_detail'),
    
    # Medical record routes
    path('patient/<int:patient_id>/record/add/', views.medical_record_add, name='medical_record_add'),
    
    # Prescription routes
    path('patient/<int:patient_id>/prescription/add/', views.prescription_add, name='prescription_add'),
    
    # Lab test routes
    path('patient/<int:patient_id>/request-lab-test/', views.request_lab_test, name='request_lab_test'),
    path('patient/<int:patient_id>/lab-test/<int:test_id>/', views.lab_test_detail, name='lab_test_detail'),
]
