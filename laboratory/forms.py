from django import forms
from .models import LabTest, LabResultItem, LabReport

class LabTestForm(forms.ModelForm):
    class Meta:
        model = LabTest
        fields = ['test_code', 'test_name', 'test_date', 'sample_type', 
                 'technician', 'ordered_by', 'priority', 'notes']
        widgets = {
            'test_date': forms.DateTimeInput(attrs={'type': 'datetime-local'}),
            'notes': forms.Textarea(attrs={'rows': 3}),
            'description': forms.Textarea(attrs={'rows': 3}),
        }

class LabResultItemForm(forms.ModelForm):
    class Meta:
        model = LabResultItem
        fields = ['test', 'result', 'normal_range', 'is_abnormal', 'notes']

class TestResultForm(forms.ModelForm):
    """Form for test results"""
    class Meta:
        model = LabTest
        fields = ['results', 'status', 'notes', 'test_name', 'sample_type', 'priority']
        widgets = {
            'results': forms.Textarea(attrs={'rows': 5, 'class': 'form-control'}),
            'status': forms.Select(attrs={'class': 'form-select'}),
            'notes': forms.Textarea(attrs={'rows': 3, 'class': 'form-control'}),
            'test_name': forms.TextInput(attrs={'class': 'form-control'}),
            'sample_type': forms.TextInput(attrs={'class': 'form-control'}),
            'priority': forms.Select(choices=[
                ('urgent', 'Urgent'),
                ('normal', 'Normal'),
                ('low', 'Low'),
            ], attrs={'class': 'form-select'})
        }

LabResultItemFormSet = forms.inlineformset_factory(
    LabTest, LabResultItem,
    form=LabResultItemForm,
    extra=3, can_delete=True
)

class LabReportUploadForm(forms.ModelForm):
    """Form for uploading lab reports"""
    class Meta:
        model = LabReport
        fields = ['report_file', 'is_final']
        widgets = {
            'report_file': forms.FileInput(attrs={'class': 'form-control'}),
        }

class TestFilterForm(forms.Form):
    """Form for filtering lab tests"""
    STATUS_CHOICES = [('', 'All Status')] + list(LabTest.TEST_STATUS_CHOICES)
    TEST_TYPE_CHOICES = [('', 'All Types')] + list(LabTest.TEST_TYPE_CHOICES)
    
    test_type = forms.ChoiceField(choices=TEST_TYPE_CHOICES, required=False,
                                widget=forms.Select(attrs={'class': 'form-select'}))
    status = forms.ChoiceField(choices=STATUS_CHOICES, required=False,
                             widget=forms.Select(attrs={'class': 'form-select'}))
    date_from = forms.DateField(required=False, 
                               widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}))
    date_to = forms.DateField(required=False, 
                             widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}))
    patient_name = forms.CharField(required=False, 
                            widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Search...'}))
