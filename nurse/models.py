from django.db import models
from django.conf import settings
from patient.models import Patient
from doctor.models import Doctor

class Nurse(models.Model):
    """Model representing a nurse"""
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='nurse_profile')
    license_number = models.CharField(max_length=50, unique=True)
    specialty = models.CharField(max_length=100, blank=True)
    years_experience = models.PositiveIntegerField(default=0)
    
    def __str__(self):
        return self.user.get_full_name()

class NurseAssignment(models.Model):
    """Model representing nurse assignments to patients"""
    nurse = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='patient_assignments')
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='nurse_assignments')
    assigned_date = models.DateField(auto_now_add=True)
    is_primary = models.BooleanField(default=False)
    notes = models.TextField(blank=True)
    
    class Meta:
        unique_together = ['nurse', 'patient']
        
    def __str__(self):
        return f"{self.nurse.get_full_name()} assigned to {self.patient.user.get_full_name()}"

class VitalSigns(models.Model):
    """Model to store patient vital signs"""
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='vitals')
    recorded_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name='recorded_vitals')
    recorded_at = models.DateTimeField(auto_now_add=True)
    
    # Vital signs
    temperature = models.DecimalField(max_digits=5, decimal_places=2, help_text="Temperature in Celsius")
    blood_pressure_systolic = models.IntegerField(help_text="Systolic blood pressure in mmHg")
    blood_pressure_diastolic = models.IntegerField(help_text="Diastolic blood pressure in mmHg")
    heart_rate = models.IntegerField(help_text="Heart rate in beats per minute")
    respiratory_rate = models.IntegerField(null=True, blank=True, help_text="Respirations per minute")
    oxygen_saturation = models.IntegerField(null=True, blank=True, help_text="Oxygen saturation in percentage")
    notes = models.TextField(blank=True, help_text="Additional notes about the vital signs")
    
    def __str__(self):
        return f"Vitals for {self.patient.user.get_full_name()} on {self.recorded_at.strftime('%Y-%m-%d %H:%M')}"
    
    @property
    def blood_pressure(self):
        """Return blood pressure as a formatted string"""
        return f"{self.blood_pressure_systolic}/{self.blood_pressure_diastolic}"
