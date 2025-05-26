from django import forms
from django.forms import inlineformset_factory
from .models import Prescription, PrescriptionItem, Medication

class PrescriptionForm(forms.ModelForm):
    """Form for creating and editing prescriptions"""
    class Meta:
        model = Prescription
        fields = ['patient', 'doctor', 'diagnosis', 'notes']
        widgets = {
            'diagnosis': forms.Textarea(attrs={'rows': 3, 'class': 'form-control'}),
            'notes': forms.Textarea(attrs={'rows': 3, 'class': 'form-control'}),
        }
    
    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        # For doctor users, prefill their own profile
        if user and user.is_doctor:
            self.fields['doctor'].initial = user.doctor_profile
            self.fields['doctor'].widget.attrs['disabled'] = True
            self.fields['doctor'].required = False

class PrescriptionItemForm(forms.ModelForm):
    """Form for prescription items"""
    class Meta:
        model = PrescriptionItem
        fields = ['medication', 'dosage', 'frequency', 'duration', 'instructions']
        widgets = {
            'instructions': forms.Textarea(attrs={'rows': 2, 'class': 'form-control'}),
        }

# Create a formset for handling multiple prescription items
PrescriptionItemFormset = inlineformset_factory(
    Prescription, 
    PrescriptionItem, 
    form=PrescriptionItemForm,
    extra=1, 
    can_delete=True
)

class MedicationForm(forms.ModelForm):
    """Form for adding/editing medications"""
    class Meta:
        model = Medication
        fields = ['name', 'description', 'dosage_form', 'strength', 'manufacturer']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3, 'class': 'form-control'}),
        }
