from django.db import models
from django.utils import timezone
from patient.models import Patient

class Payment(models.Model):
    """Model to track payments for invoices"""
    PAYMENT_METHOD_CHOICES = [
        ('online', 'Online Payment'),
        ('credit_card', 'Credit Card'),
        ('debit_card', 'Debit Card'),
        ('cash', 'Cash'),
        ('insurance', 'Insurance'),
    ]
    
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='payment_records')
    # Use string reference to avoid circular import
    invoice = models.ForeignKey('pharmacy.PrescriptionInvoice', on_delete=models.SET_NULL, 
                              null=True, related_name='payments')
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    payment_method = models.CharField(max_length=20, choices=PAYMENT_METHOD_CHOICES)
    payment_date = models.DateTimeField(default=timezone.now)
    transaction_id = models.CharField(max_length=100, blank=True)
    status = models.CharField(max_length=20, default='completed')
    notes = models.TextField(blank=True)
    
    def __str__(self):
        patient_name = self.patient.user.get_full_name() if self.patient else "Unknown"
        return f"Payment {self.id} - {patient_name} - ${self.amount}"
