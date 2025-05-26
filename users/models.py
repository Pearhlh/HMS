from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.utils.translation import gettext_lazy as _
from phonenumber_field.modelfields import PhoneNumberField


class UserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('role', User.ADMIN)
        return self.create_user(email, password, **extra_fields)


class User(AbstractUser):
    # Role choices
    PATIENT = 'patient'
    DOCTOR = 'doctor'
    NURSE = 'nurse'
    ADMIN = 'admin'
    PHARMACIST = 'pharmacist'
    INSURANCE = 'insurance'
    LAB_TECH = 'lab_technician'
    
    ROLE_CHOICES = [
        (PATIENT, 'Patient'),
        (DOCTOR, 'Doctor'),
        (NURSE, 'Nurse'),
        (ADMIN, 'Administrator'),
        (PHARMACIST, 'Pharmacist'),
        (INSURANCE, 'Insurance Provider'),
        (LAB_TECH, 'Laboratory Technician'),
    ]
    
    username = None
    email = models.EmailField(_('email address'), unique=True)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default=PATIENT)
    phone_number = PhoneNumberField(blank=True, null=True)
    address = models.TextField(blank=True, null=True)
    date_of_birth = models.DateField(null=True, blank=True)
    profile_picture = models.ImageField(upload_to='profile_pictures/', blank=True, null=True)
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []
    
    objects = UserManager()
    
    def __str__(self):
        return f"{self.email} ({self.get_role_display()})"
    
    @property
    def is_patient(self):
        return self.role == self.PATIENT
    
    @property
    def is_doctor(self):
        return self.role == self.DOCTOR
    
    @property
    def is_nurse(self):
        return self.role == self.NURSE
    
    @property
    def is_admin(self):
        return self.role == self.ADMIN
    
    @property
    def is_pharmacist(self):
        return self.role == self.PHARMACIST
    
    @property
    def is_insurance(self):
        """Check if user is an insurance provider"""
        return self.role in ['insurance', 'insurer']
    
    @property
    def is_lab_technician(self):
        return self.role == self.LAB_TECH
