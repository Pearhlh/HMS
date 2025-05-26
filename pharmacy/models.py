from django.db import models
from django.conf import settings
from django.utils import timezone
from prescription.models import Prescription, Medication, PrescriptionItem
from patient.models import Patient

class PharmacyLocation(models.Model):
    """Physical pharmacy location"""
    name = models.CharField(max_length=100)
    address = models.TextField()
    phone = models.CharField(max_length=20)
    email = models.EmailField(blank=True, null=True)
    operating_hours = models.CharField(max_length=255, help_text="E.g., Mon-Fri: 9AM-6PM, Sat: 10AM-4PM")
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return self.name


class MedicationInventory(models.Model):
    """Inventory of medications at a pharmacy location"""
    pharmacy = models.ForeignKey(PharmacyLocation, on_delete=models.CASCADE, related_name='inventory')
    medication = models.ForeignKey(Medication, on_delete=models.CASCADE, related_name='inventory')
    batch_number = models.CharField(max_length=50)
    expiry_date = models.DateField()
    quantity_in_stock = models.PositiveIntegerField()
    reorder_level = models.PositiveIntegerField(help_text="Restock when inventory falls below this level")
    unit_cost = models.DecimalField(max_digits=10, decimal_places=2)
    unit_price = models.DecimalField(max_digits=10, decimal_places=2)
    last_restocked = models.DateField(auto_now_add=True)
    
    class Meta:
        unique_together = ['pharmacy', 'medication', 'batch_number']
        verbose_name_plural = "Medication Inventories"
    
    def __str__(self):
        return f"{self.medication} at {self.pharmacy} (Batch: {self.batch_number})"
    
    @property
    def is_low_stock(self):
        return self.quantity_in_stock <= self.reorder_level
    
    @property
    def is_expired(self):
        return self.expiry_date <= timezone.now().date()


class PrescriptionDispensing(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('verified', 'Verified'),
        ('dispensed', 'Dispensed'),
        ('rejected', 'Rejected'),
    ]
    
    prescription = models.ForeignKey(Prescription, on_delete=models.CASCADE, related_name='prescription_dispensings')
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    pharmacist = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name='processed_prescriptions')
    verified_date = models.DateTimeField(null=True, blank=True)
    dispensed_date = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    notes = models.TextField(blank=True)
    
    def __str__(self):
        return f"Prescription processing for {self.patient.user.get_full_name()}"


class DispensedMedication(models.Model):
    """Details of individual medications dispensed"""
    dispensing_record = models.ForeignKey(PrescriptionDispensing, on_delete=models.CASCADE, related_name='dispensed_items')
    prescription_item = models.ForeignKey('prescription.PrescriptionItem', on_delete=models.CASCADE)
    inventory_item = models.ForeignKey(MedicationInventory, on_delete=models.CASCADE)
    quantity_dispensed = models.PositiveIntegerField()
    
    def __str__(self):
        return f"{self.prescription_item.medication.name} - {self.quantity_dispensed} units"
    
    def save(self, *args, **kwargs):
        # Update inventory quantity
        if not self.pk:  # Only on creation
            self.inventory_item.quantity_in_stock -= self.quantity_dispensed
            self.inventory_item.save()
            
        super().save(*args, **kwargs)


class MedicationDispensing(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('dispensed', 'Dispensed'),
    ]
    
    # Add a foreign key to PrescriptionDispensing
    prescription_dispensing = models.ForeignKey(PrescriptionDispensing, on_delete=models.CASCADE, related_name='medications')
    medication_name = models.CharField(max_length=100)
    dosage = models.CharField(max_length=50)
    quantity = models.PositiveIntegerField()
    pharmacist = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name='dispensed_medications')
    dispensed_date = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    notes = models.TextField(blank=True)
    
    def __str__(self):
        return f"{self.medication_name} for {self.prescription_dispensing.patient.user.get_full_name()}"


class PrescriptionInvoice(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('paid', 'Paid'),
        ('canceled', 'Canceled'),
    ]
    
    prescription_dispensing = models.OneToOneField(PrescriptionDispensing, on_delete=models.CASCADE, related_name='invoice')
    invoice_number = models.CharField(max_length=20, unique=True)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='prescription_invoices')
    pharmacist = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True)
    created_date = models.DateTimeField(auto_now_add=True)
    due_date = models.DateField()
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    payment_date = models.DateTimeField(null=True, blank=True)
    notes = models.TextField(blank=True)
    
    def __str__(self):
        return f"Invoice #{self.invoice_number} for {self.patient.user.get_full_name()}"
    
    def save(self, *args, **kwargs):
        # Generate invoice number if not provided
        if not self.invoice_number:
            last_invoice = PrescriptionInvoice.objects.order_by('-id').first()
            last_id = 0 if not last_invoice else last_invoice.id
            self.invoice_number = f"RX-{timezone.now().strftime('%Y%m')}-{last_id + 1:04d}"
        super().save(*args, **kwargs)
    
    def mark_as_paid(self):
        """Mark the invoice as paid"""
        self.status = 'paid'
        self.payment_date = timezone.now()
        self.save()
        
        # We'll create a simplified payment record system for now
        # avoiding a direct import that could create circular import issues
        try:
            # Use dynamic import to avoid circular imports
            from django.apps import apps
            Payment = apps.get_model('payment', 'Payment')
            
            # Create a payment record
            Payment.objects.create(
                patient=self.patient,
                invoice=self,
                amount=self.total_amount,
                payment_method='online',
                payment_date=timezone.now(),
                transaction_id=f"RX-PAY-{timezone.now().strftime('%Y%m%d%H%M%S')}",
                status='completed',
                notes=f"Payment for prescription invoice #{self.invoice_number}"
            )
        except Exception as e:
            print(f"Error creating payment record: {str(e)}")


class MedicationDelivery(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('in_transit', 'In Transit'),
        ('delivered', 'Delivered'),
        ('failed', 'Failed'),
    ]
    
    prescription_dispensing = models.OneToOneField(PrescriptionDispensing, on_delete=models.CASCADE, related_name='delivery')
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    delivery_address = models.TextField()
    tracking_number = models.CharField(max_length=50, blank=True)
    dispatched_date = models.DateTimeField(null=True, blank=True)
    estimated_delivery = models.DateTimeField(null=True, blank=True)
    actual_delivery = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    delivery_notes = models.TextField(blank=True)
    notification_sent = models.BooleanField(default=False)
    
    def __str__(self):
        return f"Delivery for {self.patient.user.get_full_name()}"
