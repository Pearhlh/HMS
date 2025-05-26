from django.contrib import admin
from .models import InsuranceProvider, InsurancePolicy, InsuranceClaim

@admin.register(InsuranceProvider)
class InsuranceProviderAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'phone', 'website']
    search_fields = ['name', 'email', 'phone']

@admin.register(InsurancePolicy)
class InsurancePolicyAdmin(admin.ModelAdmin):
    list_display = ['policy_number', 'provider', 'patient', 'start_date', 'end_date', 'status']
    list_filter = ['status', 'provider', 'start_date', 'end_date']
    search_fields = ['policy_number', 'patient__user__first_name', 'patient__user__last_name']

@admin.register(InsuranceClaim)
class InsuranceClaimAdmin(admin.ModelAdmin):
    list_display = ['claim_number', 'patient', 'insurance_policy', 'service_date', 'claim_amount', 'approval_status']
    list_filter = ['approval_status', 'service_date', 'claim_date']
    search_fields = ['claim_number', 'patient__user__first_name', 'patient__user__last_name', 'diagnosis_codes', 'service_codes']
