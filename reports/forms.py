from django import forms
from django.utils import timezone
from datetime import datetime, timedelta
from .models import ReportConfiguration

class ReportConfigForm(forms.ModelForm):
    """Form for creating and updating report configurations"""
    class Meta:
        model = ReportConfiguration
        fields = ['name', 'report_type', 'description', 'is_public']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3, 'class': 'form-control'}),
        }

class ReportParametersForm(forms.Form):
    """Dynamic form for report parameters based on report type"""
    def __init__(self, *args, **kwargs):
        report_type = kwargs.pop('report_type', None)
        super().__init__(*args, **kwargs)
        
        # Add common date range fields
        today = timezone.now().date()
        first_day_of_month = today.replace(day=1)
        last_month_end = first_day_of_month - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        
        self.fields['from_date'] = forms.DateField(
            initial=first_day_of_month,
            widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'})
        )
        self.fields['to_date'] = forms.DateField(
            initial=today,
            widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'})
        )
        
        # Add fields based on report type
        if report_type == 'financial':
            self.fields['payment_status'] = forms.ChoiceField(
                choices=[
                    ('', 'All Statuses'),
                    ('paid', 'Paid'),
                    ('pending', 'Pending'),
                    ('overdue', 'Overdue'),
                    ('partial', 'Partially Paid'),
                ],
                required=False,
                widget=forms.Select(attrs={'class': 'form-select'})
            )
            self.fields['include_insurance'] = forms.BooleanField(
                initial=True,
                required=False,
                widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
            )
            
        elif report_type == 'patient':
            self.fields['age_ranges'] = forms.MultipleChoiceField(
                choices=[
                    ('0-18', '0-18'),
                    ('19-35', '19-35'),
                    ('36-50', '36-50'),
                    ('51-65', '51-65'),
                    ('65+', '65+'),
                ],
                initial=['0-18', '19-35', '36-50', '51-65', '65+'],
                required=False,
                widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'})
            )
            self.fields['gender'] = forms.MultipleChoiceField(
                choices=[
                    ('M', 'Male'),
                    ('F', 'Female'),
                    ('O', 'Other'),
                ],
                initial=['M', 'F', 'O'],
                required=False,
                widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'})
            )
            
        elif report_type == 'clinical':
            self.fields['include_lab_results'] = forms.BooleanField(
                initial=True,
                required=False,
                widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
            )
            self.fields['include_prescriptions'] = forms.BooleanField(
                initial=True,
                required=False,
                widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
            )
            
        elif report_type == 'operational':
            self.fields['group_by'] = forms.ChoiceField(
                choices=[
                    ('day', 'Day'),
                    ('week', 'Week'),
                    ('month', 'Month'),
                ],
                initial='month',
                widget=forms.Select(attrs={'class': 'form-select'})
            )
            self.fields['chart_type'] = forms.ChoiceField(
                choices=[
                    ('line', 'Line Chart'),
                    ('bar', 'Bar Chart'),
                    ('pie', 'Pie Chart'),
                ],
                initial='line',
                widget=forms.Select(attrs={'class': 'form-select'})
            )
