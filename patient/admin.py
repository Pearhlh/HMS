from django.contrib import admin
from .models import Patient, MedicalRecord

@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ('user', 'gender', 'blood_type')
    list_filter = ('gender', 'blood_type')
    search_fields = ('user__email', 'user__first_name', 'user__last_name')
    
    fieldsets = (
        ('User Information', {
            'fields': ('user',)
        }),
        ('Personal Details', {
            'fields': ('gender', 'date_of_birth', 'blood_type', 'height')
        }),
        ('Medical Information', {
            'fields': ('allergies', 'chronic_diseases')
        }),
        ('Emergency Contact', {
            'fields': ('emergency_contact_name', 'emergency_contact_phone')
        }),
    )

@admin.register(MedicalRecord)
class MedicalRecordAdmin(admin.ModelAdmin):
    list_display = ('patient', 'record_date', 'diagnosis', 'created_by')
    list_filter = ('record_date',)
    search_fields = ('patient__user__first_name', 'patient__user__last_name', 'diagnosis', 'treatment')
    date_hierarchy = 'record_date'
