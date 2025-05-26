from django.db import models
from django.utils import timezone
from django.conf import settings

class InsuranceProvider(models.Model):
    """Insurance company providing coverage"""
    name = models.CharField(max_length=100)
    address = models.TextField()
    phone = models.CharField(max_length=20)
    email = models.EmailField()
    website = models.URLField(blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return self.name
    
    @property
    def active_policies_count(self):
        return self.policies.filter(status='active').count()


class InsurancePolicy(models.Model):
    """Insurance policy details for a patient"""
    STATUS_CHOICES = [
        ('active', 'Active'),
        ('expired', 'Expired'),
        ('canceled', 'Canceled'),
        ('pending', 'Pending Approval'),
    ]
    
    # Using string reference to avoid circular import
    patient = models.ForeignKey('patient.Patient', on_delete=models.CASCADE, related_name='insurance_policies')
    provider = models.ForeignKey(InsuranceProvider, on_delete=models.CASCADE, related_name='policies')
    policy_number = models.CharField(max_length=50, unique=True)
    member_id = models.CharField(max_length=50)
    group_number = models.CharField(max_length=50, blank=True, null=True)
    start_date = models.DateField()
    end_date = models.DateField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')
    coverage_percentage = models.IntegerField(help_text="Coverage percentage (0-100)")
    coverage_details = models.TextField(help_text="Details about what is covered")
    deductible = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    co_pay = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    out_of_pocket_max = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.patient.user.get_full_name()} - {self.policy_number}"
    
    @property
    def is_valid(self):
        today = timezone.now().date()
        return self.status == 'active' and self.start_date <= today <= self.end_date
    
    def save(self, *args, **kwargs):
        # Remove the problematic code that tries to update the patient's reference
        # Instead, do nothing special when status is active
        # Just save the policy itself
        super().save(*args, **kwargs)


class InsuranceClaim(models.Model):
    """Insurance claim submitted for medical services"""
    APPROVAL_STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('approved', 'Approved'),
        ('partial', 'Partially Approved'),
        ('rejected', 'Rejected'),
        ('canceled', 'Canceled'),
    ]
    
    claim_number = models.CharField(max_length=50, unique=True)
    # Using string reference to avoid circular import
    patient = models.ForeignKey('patient.Patient', on_delete=models.CASCADE, related_name='insurance_claims')
    insurance_policy = models.ForeignKey(InsurancePolicy, on_delete=models.CASCADE, related_name='claims')
    service_date = models.DateField()
    claim_date = models.DateField(auto_now_add=True)
    claim_amount = models.DecimalField(max_digits=10, decimal_places=2)
    approved_amount = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    approval_status = models.CharField(max_length=20, choices=APPROVAL_STATUS_CHOICES, default='pending')
    diagnosis_codes = models.CharField(max_length=255, help_text="Comma-separated diagnosis codes")
    service_codes = models.CharField(max_length=255, help_text="Comma-separated service codes")
    notes = models.TextField(blank=True, null=True)
    processed_date = models.DateTimeField(null=True, blank=True)
    processed_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, 
                                    null=True, blank=True, related_name='processed_claims')
    
    def __str__(self):
        return f"Claim {self.claim_number} - {self.patient.user.get_full_name()}"
    
    def save(self, *args, **kwargs):
        # Generate claim number if not provided
        if not self.claim_number:
            # Format: CL-YYYY-RANDOM
            import random
            import string
            from django.utils import timezone
            
            year = timezone.now().year
            random_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            self.claim_number = f"CL-{year}-{random_part}"
        
        # Ensure new claims have pending status
        if not self.pk and not self.approval_status:
            self.approval_status = 'pending'
            
        super().save(*args, **kwargs)
