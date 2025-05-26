from django.contrib import admin
from .models import Payment

@admin.register(Payment)
class PaymentAdmin(admin.ModelAdmin):
    list_display = ['id', 'patient', 'amount', 'payment_method', 'payment_date', 'status']
    list_filter = ['payment_method', 'status', 'payment_date']
    search_fields = ['patient__user__first_name', 'patient__user__last_name', 'transaction_id']
    date_hierarchy = 'payment_date'
