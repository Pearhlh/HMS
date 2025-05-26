from django.contrib import admin
from .models import Doctor, Specialization
from appointment.models import Department

@admin.register(Specialization)
class SpecializationAdmin(admin.ModelAdmin):
    list_display = ('name', 'description')
    search_fields = ('name', 'description')

@admin.register(Doctor)
class DoctorAdmin(admin.ModelAdmin):
    list_display = ('user', 'specialization', 'department', 'accepting_patients')
    list_filter = ('accepting_patients', 'department')
    search_fields = ('user__first_name', 'user__last_name', 'specialization')
    fieldsets = (
        ('User Information', {
            'fields': ('user',)
        }),
        ('Professional Details', {
            'fields': ('specialization', 'license_number', 'experience_years', 'department', 'consultation_fee')
        }),
        ('Availability', {
            'fields': ('accepting_patients', 'available_days', 'available_hours_start', 'available_hours_end')
        }),
        ('Additional Information', {
            'fields': ('bio', 'qualifications')
        }),
    )
