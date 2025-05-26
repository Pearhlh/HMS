from django import forms
from django.contrib.auth import get_user_model
from allauth.account.forms import SignupForm

User = get_user_model()

class CustomUserCreationForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ('email', 'first_name', 'last_name')

class CustomUserChangeForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ('email', 'first_name', 'last_name')

class CustomSignupForm(SignupForm):
    """Custom signup form that properly inherits from allauth's SignupForm"""
    ROLE_CHOICES = [
        ('patient', 'Patient'),
        ('doctor', 'Doctor'),
        ('nurse', 'Nurse'),
        ('lab_tech', 'Laboratory Technician'),
        ('pharmacy', 'Pharmacist'),
        ('insurer', 'Insurance Provider'),
        ('admin', 'Administrator'),
    ]
    
    # Add our custom fields
    first_name = forms.CharField(max_length=30, label='First Name')
    last_name = forms.CharField(max_length=30, label='Last Name')
    role = forms.ChoiceField(choices=ROLE_CHOICES, required=True)
    
    # We don't need to override save here as the adapter will handle it
