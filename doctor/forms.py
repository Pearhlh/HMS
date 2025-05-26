from django import forms
from django.contrib.auth.forms import UserChangeForm
from django.contrib.auth import get_user_model
from .models import Doctor, Specialization

User = get_user_model()

class DoctorForm(forms.ModelForm):
    """Form for creating and updating doctor profiles"""
    class Meta:
        model = Doctor
        fields = [
            'specialization', 
            'license_number', 
            'experience_years', 
            'department', 
            'bio', 
            'accepting_patients',
            'consultation_fee',
            'qualifications',
            'available_days',
            'available_hours_start',
            'available_hours_end'
        ]
        widgets = {
            'bio': forms.Textarea(attrs={'rows': 3}),
            'qualifications': forms.Textarea(attrs={'rows': 3}),
            'available_hours_start': forms.TimeInput(attrs={'type': 'time'}),
            'available_hours_end': forms.TimeInput(attrs={'type': 'time'}),
            'available_days': forms.TextInput(attrs={'placeholder': 'monday,tuesday,wednesday'})
        }

class DoctorUserForm(UserChangeForm):
    """Form for doctor user information"""
    password = None  # Don't show password field
    
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email']
        widgets = {
            'first_name': forms.TextInput(attrs={'class': 'form-control'}),
            'last_name': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
        }

class SpecializationForm(forms.ModelForm):
    """Form for specializations"""
    class Meta:
        model = Specialization
        fields = ['name', 'description']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3, 'class': 'form-control'}),
        }
