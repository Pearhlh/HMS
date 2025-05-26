from django.db import models
from django.conf import settings
from django.contrib.postgres.fields import ArrayField
from django.utils.translation import gettext_lazy as _

class Specialization(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return self.name

class Doctor(models.Model):
    DAYS_OF_WEEK = [
        ('monday', 'Monday'),
        ('tuesday', 'Tuesday'),
        ('wednesday', 'Wednesday'),
        ('thursday', 'Thursday'),
        ('friday', 'Friday'),
        ('saturday', 'Saturday'),
        ('sunday', 'Sunday'),
    ]

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        related_name='doctor_profile'
    )
    specialization = models.CharField(max_length=100)
    license_number = models.CharField(max_length=50)
    experience_years = models.PositiveIntegerField(default=0)
    department = models.ForeignKey(
        'appointment.Department',
        on_delete=models.SET_NULL,
        null=True,
        related_name='doctors'
    )
    bio = models.TextField(blank=True, null=True)
    accepting_patients = models.BooleanField(default=True)
    
    # Additional fields that were missing
    consultation_fee = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    qualifications = models.TextField(blank=True)
    available_days = models.CharField(max_length=100, blank=True, help_text="Comma-separated days")
    available_hours_start = models.TimeField(null=True, blank=True)
    available_hours_end = models.TimeField(null=True, blank=True)
    
    def __str__(self):
        return f"Dr. {self.user.get_full_name()}"
    
    @property
    def available_days_list(self):
        """Return available days as a list"""
        if self.available_days:
            return [day.strip() for day in self.available_days.split(',')]
        return []
