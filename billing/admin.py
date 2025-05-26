from django.contrib import admin
from .models import Bill, Payment

@admin.register(Bill)
class BillAdmin(admin.ModelAdmin):
    list_display = ['invoice_number', 'patient', 'amount', 'due_date', 'status', 'created_date']
    list_filter = ['status', 'due_date', 'created_date']
    search_fields = ['invoice_number', 'patient__user__first_name', 'patient__user__last_name']

@admin.register(Payment)
class PaymentAdmin(admin.ModelAdmin):
    list_display = ['bill', 'amount', 'payment_date', 'payment_method', 'status']
    list_filter = ['status', 'payment_method', 'payment_date']
    search_fields = ['bill__invoice_number', 'bill__patient__user__first_name', 'bill__patient__user__last_name']
