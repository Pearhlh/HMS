from django.db import models
from django.utils import timezone
from django.conf import settings
from appointment.models import Appointment
from patient.models import Patient

class Bill(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('paid', 'Paid'),
        ('overdue', 'Overdue'),
        ('cancelled', 'Cancelled'),
    ]
    
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='bills')
    invoice_number = models.CharField(max_length=20, unique=True)
    description = models.TextField()
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    created_date = models.DateTimeField(default=timezone.now)
    due_date = models.DateField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    paid_date = models.DateTimeField(null=True, blank=True)
    paid_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    
    def __str__(self):
        return f"Bill #{self.invoice_number} - {self.patient.user.get_full_name()}"
    
    def mark_as_paid(self):
        self.status = 'paid'
        self.paid_date = timezone.now()
        self.paid_amount = self.amount
        self.save()
    
    @property
    def is_overdue(self):
        return self.status == 'pending' and self.due_date < timezone.now().date()
    
    def save(self, *args, **kwargs):
        # Auto update status to overdue if past due date
        if self.status == 'pending' and self.due_date < timezone.now().date():
            self.status = 'overdue'
        super().save(*args, **kwargs)


class BillItem(models.Model):
    bill = models.ForeignKey(Bill, on_delete=models.CASCADE, related_name='items')
    description = models.CharField(max_length=255)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    quantity = models.PositiveIntegerField(default=1)
    
    def __str__(self):
        return f"{self.description} - {self.amount}"
    
    @property
    def total(self):
        return self.amount * self.quantity


class PaymentMethod(models.Model):
    TYPE_CHOICES = [
        ('credit_card', 'Credit Card'),
        ('bank_account', 'Bank Account'),
    ]
    
    CARD_BRAND_CHOICES = [
        ('visa', 'Visa'),
        ('mastercard', 'MasterCard'),
        ('amex', 'American Express'),
        ('discover', 'Discover'),
        ('other', 'Other'),
    ]
    
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='payment_methods')
    type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    is_default = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Credit card fields
    card_brand = models.CharField(max_length=20, choices=CARD_BRAND_CHOICES, null=True, blank=True)
    card_last_four = models.CharField(max_length=4, null=True, blank=True)
    expiry_month = models.CharField(max_length=2, null=True, blank=True)
    expiry_year = models.CharField(max_length=4, null=True, blank=True)
    
    # Bank account fields
    bank_name = models.CharField(max_length=100, null=True, blank=True)
    account_last_four = models.CharField(max_length=4, null=True, blank=True)
    account_type = models.CharField(max_length=20, null=True, blank=True, 
                                   choices=[
                                       ('checking', 'Checking'),
                                       ('savings', 'Savings'),
                                   ])
    
    def __str__(self):
        if self.type == 'credit_card':
            return f"{self.get_card_brand_display()} •••• {self.card_last_four}"
        else:
            return f"{self.bank_name} •••• {self.account_last_four}"


class Payment(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('refunded', 'Refunded'),
    ]
    
    PAYMENT_METHOD_CHOICES = [
        ('credit_card', 'Credit Card'),
        ('bank_transfer', 'Bank Transfer'),
        ('cash', 'Cash'),
        ('insurance', 'Insurance'),
    ]
    
    bill = models.ForeignKey(Bill, on_delete=models.CASCADE, related_name='payments')
    patient = models.ForeignKey('patient.Patient', on_delete=models.CASCADE, related_name='payments')
    payment_id = models.CharField(max_length=100, unique=True)
    payment_date = models.DateTimeField(default=timezone.now)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    payment_method = models.CharField(max_length=20, choices=PAYMENT_METHOD_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # For credit card payments
    card_last_four = models.CharField(max_length=4, null=True, blank=True)
    
    # For tracking reference
    transaction_id = models.CharField(max_length=100, null=True, blank=True)
    description = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return f"Payment #{self.payment_id} - {self.amount}"

    def save(self, *args, **kwargs):
        is_new = self.pk is None
        super().save(*args, **kwargs)
        
        if is_new and self.status == 'completed':
            # Update the bill amount paid
            self.bill.paid_amount += self.amount
            if self.bill.paid_amount >= self.bill.amount:
                self.bill.status = 'paid'
            else:
                self.bill.status = 'pending'
            self.bill.save()
