from django import forms
from .models import InsuranceProvider, InsurancePolicy, InsuranceClaim

class InsuranceProviderForm(forms.ModelForm):
    class Meta:
        model = InsuranceProvider
        fields = ['name', 'address', 'phone', 'email', 'website', 'description']
        widgets = {
            'address': forms.Textarea(attrs={'rows': 3, 'class': 'form-control'}),
            'description': forms.Textarea(attrs={'rows': 3, 'class': 'form-control'}),
        }

class InsurancePolicyForm(forms.ModelForm):
    class Meta:
        model = InsurancePolicy
        fields = ['patient', 'provider', 'policy_number', 'member_id', 'group_number',
                 'start_date', 'end_date', 'status', 'coverage_percentage', 
                 'coverage_details', 'deductible', 'co_pay', 'out_of_pocket_max']
        widgets = {
            'start_date': forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
            'end_date': forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
            'coverage_details': forms.Textarea(attrs={'rows': 3, 'class': 'form-control'}),
        }
    
    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        # Check if there are any providers
        provider_count = self.fields['provider'].queryset.count()
        if provider_count == 0:
            self.fields['provider'].help_text = "No insurance providers are available. Please contact administration."
        else:
            self.fields['provider'].help_text = f"{provider_count} providers available"
        
        # For patient users, prefill their own profile
        if user and user.is_patient:
            try:
                from patient.models import Patient
                patient = Patient.objects.get(user=user)
                self.fields['patient'].initial = patient
                self.fields['patient'].widget.attrs['disabled'] = True
                self.fields['patient'].required = False
                
                # Hide status field from patients - always set to 'pending' for new submissions
                if not kwargs.get('instance'):
                    self.fields['status'].initial = 'pending'
                    self.fields['status'].widget = forms.HiddenInput()
            except:
                pass
                
    def clean_patient(self):
        # Return the initial value if field is disabled to prevent it from being blank
        if self.fields['patient'].widget.attrs.get('disabled'):
            return self.initial.get('patient') 
        return self.cleaned_data.get('patient')

class InsuranceClaimForm(forms.ModelForm):
    class Meta:
        model = InsuranceClaim
        fields = ['patient', 'insurance_policy', 'service_date', 'claim_amount',
                 'diagnosis_codes', 'service_codes', 'notes']
        widgets = {
            'service_date': forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
            'notes': forms.Textarea(attrs={'rows': 3, 'class': 'form-control'}),
        }
    
    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        # For patient users, limit to their own insurance policies
        if user and user.is_patient:
            try:
                from patient.models import Patient
                patient = Patient.objects.get(user=user)
                self.fields['patient'].initial = patient
                self.fields['patient'].widget.attrs['disabled'] = True
                self.fields['patient'].required = False
                self.fields['insurance_policy'].queryset = InsurancePolicy.objects.filter(patient=patient)
            except:
                pass
        
        # Show only active insurance policies
        self.fields['insurance_policy'].queryset = InsurancePolicy.objects.filter(status='active')
