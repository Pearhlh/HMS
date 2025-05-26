from django.db import models
from django.conf import settings

class ReportSchedule(models.Model):
    """Model to schedule automatic reports"""
    FREQUENCY_CHOICES = [
        ('daily', 'Daily'),
        ('weekly', 'Weekly'),
        ('monthly', 'Monthly'),
        ('quarterly', 'Quarterly'),
    ]
    
    REPORT_TYPES = [
        ('patient_visits', 'Patient Visits'),
        ('revenue', 'Revenue Report'),
        ('lab_tests', 'Laboratory Tests'),
        ('prescriptions', 'Prescriptions'),
        ('insurance_claims', 'Insurance Claims'),
    ]
    
    title = models.CharField(max_length=100)
    report_type = models.CharField(max_length=50, choices=REPORT_TYPES)
    frequency = models.CharField(max_length=10, choices=FREQUENCY_CHOICES)
    recipients = models.TextField(help_text="Comma-separated email addresses")
    is_active = models.BooleanField(default=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    last_run = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.title} ({self.get_frequency_display()})"


class ReportConfiguration(models.Model):
    """Model for saved report configurations"""
    REPORT_TYPES = [
        ('patient', 'Patient Reports'),
        ('financial', 'Financial Reports'),
        ('clinical', 'Clinical Reports'),
        ('operational', 'Operational Reports'),
        ('pharmacy', 'Pharmacy Reports'),
        ('laboratory', 'Laboratory Reports'),
        ('insurance', 'Insurance Reports'),
    ]
    
    name = models.CharField(max_length=100)
    report_type = models.CharField(max_length=20, choices=REPORT_TYPES)
    description = models.TextField(blank=True, null=True)
    parameters = models.JSONField(help_text="Report parameters as JSON")
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='report_configurations')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_public = models.BooleanField(default=False, help_text="Whether this configuration can be used by other users")
    
    def __str__(self):
        return f"{self.name} ({self.get_report_type_display()})"


class GeneratedReport(models.Model):
    """Model for storing generated report results"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('generating', 'Generating'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    configuration = models.ForeignKey(ReportConfiguration, on_delete=models.SET_NULL, null=True, related_name='generated_reports')
    name = models.CharField(max_length=100)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    parameters_used = models.JSONField(help_text="Parameters used to generate this report")
    result_data = models.JSONField(null=True, blank=True, help_text="Report results as JSON")
    file_path = models.CharField(max_length=255, blank=True, null=True, help_text="Path to generated file if applicable")
    file_size = models.PositiveIntegerField(null=True, blank=True, help_text="Size of generated file in bytes")
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='generated_reports')
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.name} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"
