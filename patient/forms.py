from django import forms
from django.contrib.auth.forms import UserChangeForm
from .models import Patient
from django.contrib.auth import get_user_model

User = get_user_model()

class PatientUserForm(forms.ModelForm):
    """Form for updating User-related fields of a Patient"""
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email']

class PatientForm(forms.ModelForm):
    """Form for updating Patient-specific fields"""
    class Meta:
        model = Patient
        fields = [
            'date_of_birth', 'gender', 
            'blood_type', 'allergies',
        ]
        widgets = {
            'date_of_birth': forms.DateInput(attrs={'type': 'date'}),
            'allergies': forms.Textarea(attrs={'rows': 3}),
        }

class MedicalRecordForm(forms.Form):
    # A simple form for requesting medical records
    start_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))
    end_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))
    record_type = forms.ChoiceField(choices=[
        ('all', 'All Records'),
        ('appointment', 'Appointments'),
        ('prescription', 'Prescriptions'),
        ('lab_result', 'Lab Results')
    ])
