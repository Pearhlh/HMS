from django.db import models
from django.conf import settings
from patient.models import Patient
from doctor.models import Doctor

class Medication(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    dosage_form = models.CharField(max_length=50)  # e.g., tablet, capsule, liquid
    strength = models.CharField(max_length=50)  # e.g., 10mg, 500mg
    manufacturer = models.CharField(max_length=100, blank=True, null=True)
    
    def __str__(self):
        return f"{self.name} {self.strength} {self.dosage_form}"

class Prescription(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('verified', 'Verified'),
        ('dispensed', 'Dispensed'),
        ('completed', 'Completed'),
        ('rejected', 'Rejected'),
    ]
    
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='prescriptions')
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE, related_name='prescriptions')
    date_prescribed = models.DateTimeField(auto_now_add=True)
    diagnosis = models.TextField()
    notes = models.TextField(blank=True, null=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    def __str__(self):
        return f"Prescription for {self.patient} by Dr. {self.doctor.user.last_name} on {self.date_prescribed.strftime('%Y-%m-%d')}"

class PrescriptionItem(models.Model):
    prescription = models.ForeignKey(Prescription, on_delete=models.CASCADE, related_name='items')
    medication = models.ForeignKey(Medication, on_delete=models.CASCADE)
    dosage = models.CharField(max_length=100)  # e.g., "1 tablet"
    frequency = models.CharField(max_length=100)  # e.g., "twice daily"
    duration = models.CharField(max_length=100)  # e.g., "7 days"
    instructions = models.TextField(blank=True, null=True)  # e.g., "Take with food"
    
    def __str__(self):
        return f"{self.medication.name} - {self.dosage} {self.frequency} for {self.duration}"
