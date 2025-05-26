from django import forms
from django.utils import timezone
from datetime import timedelta
from .models import Appointment
from doctor.models import Doctor

class AppointmentForm(forms.ModelForm):
    """Form for creating and editing appointments"""
    
    class Meta:
        model = Appointment
        fields = ['patient', 'doctor', 'appointment_date', 'appointment_time', 'reason']
        widgets = {
            'appointment_date': forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
            'appointment_time': forms.TimeInput(attrs={'type': 'time', 'class': 'form-control'}),
            'reason': forms.Textarea(attrs={'rows': 3, 'class': 'form-control'}),
        }
    
    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        # For patient users, prefill their own patient profile
        if user and user.is_patient:
            self.fields['patient'].initial = user.patient_profile
            self.fields['patient'].widget.attrs['disabled'] = True
            self.fields['patient'].required = False
        
        # Only show active doctors
        self.fields['doctor'].queryset = Doctor.objects.filter(user__is_active=True)

    def clean(self):
        cleaned_data = super().clean()
        appointment_date = cleaned_data.get('appointment_date')
        appointment_time = cleaned_data.get('appointment_time')
        doctor = cleaned_data.get('doctor')
        
        # Check if date is in the future
        if appointment_date:
            today = timezone.now().date()
            if appointment_date < today:
                self.add_error('appointment_date', 'Appointment date cannot be in the past')
        
        # Check if doctor is available at this time
        if appointment_date and appointment_time and doctor:
            # Check if the day is in doctor's available days
            day_name = appointment_date.strftime('%A')
            if day_name not in doctor.available_days:
                self.add_error('appointment_date', f'Doctor is not available on {day_name}')
            
            # Check if time is within doctor's available hours
            if appointment_time < doctor.available_hours_start.replace(tzinfo=None) or \
               appointment_time > doctor.available_hours_end.replace(tzinfo=None):
                self.add_error('appointment_time', 'Appointment time is outside doctor\'s available hours')
            
            # Check for overlapping appointments
            existing_appointments = Appointment.objects.filter(
                doctor=doctor,
                appointment_date=appointment_date,
                status__in=['scheduled', 'confirmed']
            )
            
            for existing in existing_appointments:
                existing_time = existing.appointment_time.replace(tzinfo=None)
                # Assuming 30-minute appointments
                appointment_end_time = (
                    timezone.datetime.combine(timezone.datetime.today(), appointment_time) + 
                    timedelta(minutes=30)
                ).time()
                
                existing_end_time = (
                    timezone.datetime.combine(timezone.datetime.today(), existing_time) + 
                    timedelta(minutes=30)
                ).time()
                
                # Check for overlap
                if (appointment_time <= existing_end_time and 
                    appointment_end_time >= existing_time and
                    existing.id != self.instance.id):
                    self.add_error('appointment_time', 'This time slot is already booked')
        
        return cleaned_data

class AppointmentStatusForm(forms.Form):
    """Form for updating appointment status"""
    STATUS_CHOICES = [
        ('scheduled', 'Scheduled'),
        ('confirmed', 'Confirmed'),
        ('completed', 'Completed'),
        ('canceled', 'Canceled'),
        ('no_show', 'No Show'),
    ]
    
    status = forms.ChoiceField(
        choices=STATUS_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    notes = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={'rows': 3, 'class': 'form-control'})
    )
