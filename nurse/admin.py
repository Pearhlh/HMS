from django.contrib import admin
from .models import VitalSigns, Nurse, NurseAssignment

@admin.register(VitalSigns)
class VitalSignsAdmin(admin.ModelAdmin):
    list_display = ['patient', 'recorded_by', 'recorded_at', 'temperature', 'blood_pressure', 'heart_rate']
    list_filter = ['recorded_at', 'recorded_by']
    search_fields = ['patient__user__first_name', 'patient__user__last_name']
    date_hierarchy = 'recorded_at'

@admin.register(Nurse)
class NurseAdmin(admin.ModelAdmin):
    list_display = ['user', 'license_number', 'specialty', 'years_experience']
    search_fields = ['user__first_name', 'user__last_name', 'license_number']

@admin.register(NurseAssignment)
class NurseAssignmentAdmin(admin.ModelAdmin):
    list_display = ['nurse', 'patient', 'assigned_date', 'is_primary']
    list_filter = ['assigned_date', 'is_primary']
    search_fields = ['nurse__first_name', 'nurse__last_name', 'patient__user__first_name', 'patient__user__last_name']
