from django.urls import path
from . import views

urlpatterns = [
    path('invoices/', views.invoice_list, name='invoice_list'),
    path('invoices/<int:pk>/', views.invoice_detail, name='invoice_detail'),
    path('invoices/<int:pk>/payment/', views.process_payment, name='process_payment'),
    path('invoices/create/', views.create_invoice, name='create_invoice'),
    path('reports/', views.financial_reports, name='financial_reports'),
]
