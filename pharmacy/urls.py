from django.urls import path
from . import views

app_name = 'pharmacy'

urlpatterns = [
    # Dashboard
    path('', views.dashboard, name='dashboard'),

    # Prescription routes
    path('prescriptions/', views.prescription_list, name='prescription_list'),
    path('prescriptions/<int:prescription_id>/process/', views.process_prescription, name='process_prescription'),
    path('prescriptions/<int:pk>/dispense/', views.dispense_prescription, name='dispense_prescription'),
    path('prescriptions/pending/', views.pharmacy_prescriptions, name='pharmacy_prescriptions'),

    # Inventory routes
    path('inventory/', views.inventory_list, name='inventory_list'),
    path('inventory/add/', views.add_inventory, name='add_inventory'),

    # Invoice routes
    path('dispensing/<int:dispensing_id>/invoice/create/', views.create_invoice, name='create_invoice'),
    path('invoices/<int:invoice_id>/', views.view_invoice, name='view_invoice'),

    # Delivery routes
    path('dispensing/<int:dispensing_id>/delivery/', views.manage_delivery, name='manage_delivery'),
    path('deliveries/<int:delivery_id>/', views.view_delivery, name='view_delivery'),
]