from django.urls import path
from . import views

app_name = 'patient'

urlpatterns = [
    path('dashboard/', views.dashboard, name='dashboard'),
    path('appointments/', views.appointment_list, name='appointments'),
    path('appointment/book/', views.appointment_book, name='appointment_book'),
    path('appointment/<int:pk>/', views.appointment_detail, name='appointment_detail'),
    path('appointment/<int:pk>/cancel/', views.appointment_cancel, name='appointment_cancel'),
    path('appointment/<int:pk>/reschedule/', views.appointment_reschedule, name='appointment_reschedule'),
    path('medical-records/', views.medical_record_list, name='medical_records'),
    path('medical-record/<int:pk>/', views.medical_record_detail, name='medical_record_detail'),
    path('medical-record/<int:pk>/pdf/', views.medical_record_pdf, name='medical_record_pdf'),
    path('prescriptions/', views.prescription_list, name='prescriptions'),
    path('prescription/<int:pk>/', views.prescription_detail, name='prescription_detail'),
    
    # Bills and payments
    path('bills/', views.bills, name='bills'),
    path('bill/<int:pk>/', views.bill_detail, name='bill_detail'),
    path('bill/<int:pk>/pay/', views.bill_pay, name='bill_pay'),
    path('notifications/', views.notifications, name='notifications'),
    path('notifications/<int:pk>/', views.notification_detail, name='notification_detail'),
    path('profile/', views.profile_view, name='profile'),
    path('profile/edit/', views.profile_edit, name='profile_edit'),
    path('profile/insurance/', views.submit_insurance, name='submit_insurance'),
    path('payment-methods/', views.payment_methods, name='payment_methods'),
    
    # Lab tests
    path('lab-test/<int:pk>/', views.lab_test_detail, name='lab_test_detail'),
    path('lab-results/<int:lab_result_id>/', views.view_lab_result, name='view_lab_result'),
    
    # Invoice routes
    path('invoice/<int:pk>/', views.invoice_detail, name='invoice_detail'),

    # Vitals
    path('vitals/<int:vitals_id>/', views.view_vitals, name='view_vitals'),

    # Policies
    path('policies/', views.patient_policies, name='policies'),

    # Health records
    path('health-records/', views.health_records, name='health_records'),

    # Notifications check
    path('check-notifications/', views.check_notifications, name='check_notifications'),

    # Chat with AI
    path('chat/', views.chat_with_ai, name='chat'),
]
