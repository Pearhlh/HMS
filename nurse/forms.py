from django import forms
from .models import VitalSigns, NurseAssignment

class VitalSignsForm(forms.ModelForm):
    """Form for recording patient vital signs"""
    care_note = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 3}),
        required=False,
        help_text="Optional care note to be shared with patient and physician"
    )
    
    class Meta:
        model = VitalSigns
        fields = [
            'temperature',
            'blood_pressure_systolic',
            'blood_pressure_diastolic',
            'heart_rate',
            'respiratory_rate',
            'oxygen_saturation',
            'notes'
        ]
        widgets = {
            'notes': forms.Textarea(attrs={'rows': 3}),
        }

class NurseAssignmentForm(forms.ModelForm):
    """Form for assigning nurses to patients"""
    class Meta:
        model = NurseAssignment
        fields = ['patient', 'is_primary', 'notes']
        widgets = {
            'notes': forms.Textarea(attrs={'rows': 3}),
        }
