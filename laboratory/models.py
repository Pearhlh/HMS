from django.db import models
from django.utils import timezone
from django.conf import settings
from patient.models import Patient
from doctor.models import Doctor

class LabTest(models.Model):
    TEST_STATUS_CHOICES = [
        ('requested', 'Requested'),
        ('scheduled', 'Scheduled'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('canceled', 'Canceled'),
    ]
    
    TEST_TYPE_CHOICES = [
        ('general', 'General'),
        ('blood', 'Blood'),
        ('urine', 'Urine'),
        ('imaging', 'Imaging'),
        ('other', 'Other')
    ]
    
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='lab_tests')
    requested_by = models.ForeignKey(Doctor, on_delete=models.SET_NULL, null=True, related_name='requested_tests')
    assigned_to = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name='assigned_tests')
    test_type = models.CharField(
        max_length=50, 
        default='general',
        choices=TEST_TYPE_CHOICES
    )
    test_code = models.CharField(max_length=50, unique=True, null=True, blank=True)
    description = models.TextField(blank=True, null=True)
    instructions = models.TextField(blank=True, null=True)
    requested_date = models.DateTimeField(default=timezone.now)
    scheduled_date = models.DateTimeField(null=True, blank=True)
    completed_date = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=TEST_STATUS_CHOICES, default='requested')
    results = models.TextField(blank=True)
    test_name = models.CharField(max_length=100, null=True, blank=True)
    test_date = models.DateTimeField(null=True, blank=True)
    sample_type = models.CharField(max_length=50, null=True, blank=True)
    technician = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        related_name='lab_tests',
        null=True, blank=True
    )
    ordered_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        related_name='ordered_tests',
        null=True, blank=True
    )
    priority = models.CharField(max_length=20, choices=[
        ('urgent', 'Urgent'),
        ('normal', 'Normal'),
        ('low', 'Low'),
    ], default='normal')
    notes = models.TextField(blank=True, null=True)
    
    def __str__(self):
        if self.test_name:
            return self.test_name
        return f"Test #{self.id}"
    
    @property
    def name(self):
        """Property for backward compatibility"""
        return self.test_name or f"Test #{self.id}"


class TestResult(models.Model):
    """Model for test results"""
    lab_test = models.ForeignKey(LabTest, on_delete=models.CASCADE, related_name='test_results')
    result_value = models.CharField(max_length=255)
    reference_range = models.CharField(max_length=100, blank=True, null=True)
    is_abnormal = models.BooleanField(default=False)
    notes = models.TextField(blank=True, null=True)
    uploaded_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Result for {self.lab_test}"


class LabReport(models.Model):
    """Model for generated lab reports"""
    lab_test = models.ForeignKey(LabTest, on_delete=models.CASCADE, related_name='reports')
    report_file = models.FileField(upload_to='lab_reports/')
    uploaded_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    is_final = models.BooleanField(default=False)
    
    def __str__(self):
        return f"Report for {self.lab_test}"


class LabResultItem(models.Model):
    test = models.ForeignKey('LabTest', on_delete=models.CASCADE, related_name='result_items')
    result = models.CharField(max_length=100)
    normal_range = models.CharField(max_length=100, blank=True, null=True)
    is_abnormal = models.BooleanField(default=False)
    notes = models.TextField(blank=True, null=True)
    parameter_name = models.CharField(max_length=100)
    unit = models.CharField(max_length=50, blank=True)
    reference_range = models.CharField(max_length=100, blank=True)
    
    def __str__(self):
        return f"{self.test.test_name}: {self.result}"
