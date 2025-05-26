from django.contrib import admin
from .models import Medication, Prescription, PrescriptionItem

@admin.register(Medication)
class MedicationAdmin(admin.ModelAdmin):
    list_display = ('name', 'dosage_form', 'strength', 'manufacturer')
    search_fields = ('name', 'description', 'manufacturer')
    list_filter = ('dosage_form',)

class PrescriptionItemInline(admin.TabularInline):
    model = PrescriptionItem
    extra = 1

@admin.register(Prescription)
class PrescriptionAdmin(admin.ModelAdmin):
    list_display = ('id', 'patient', 'doctor', 'date_prescribed', 'status')
    list_filter = ('status', 'date_prescribed')
    search_fields = ('patient__user__first_name', 'patient__user__last_name', 
                     'doctor__user__first_name', 'doctor__user__last_name',
                     'diagnosis')
    date_hierarchy = 'date_prescribed'
    inlines = [PrescriptionItemInline]
