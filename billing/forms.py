from django import forms
from .models import Invoice, InvoiceItem, Payment

class InvoiceForm(forms.ModelForm):
    class Meta:
        model = Invoice
        fields = ['patient', 'appointment', 'due_date', 'total_amount', 
                 'insurance_amount', 'patient_responsibility', 'notes']
        widgets = {
            'due_date': forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
            'notes': forms.Textarea(attrs={'rows': 3, 'class': 'form-control'}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Set default due date to 30 days from today
        if not self.instance.pk:
            from datetime import date, timedelta
            self.fields['due_date'].initial = date.today() + timedelta(days=30)
    
    def clean(self):
        cleaned_data = super().clean()
        total_amount = cleaned_data.get('total_amount')
        insurance_amount = cleaned_data.get('insurance_amount')
        patient_responsibility = cleaned_data.get('patient_responsibility')
        
        # Ensure patient_responsibility + insurance_amount = total_amount
        if total_amount and insurance_amount is not None and patient_responsibility is not None:
            if abs((insurance_amount + patient_responsibility) - total_amount) > 0.01:
                raise forms.ValidationError(
                    "Insurance amount plus patient responsibility must equal the total amount"
                )
        
        return cleaned_data

class InvoiceItemForm(forms.ModelForm):
    class Meta:
        model = InvoiceItem
        fields = ['item_type', 'description', 'quantity', 'unit_price', 'discount']
        widgets = {
            'description': forms.TextInput(attrs={'class': 'form-control'}),
        }
    
    def clean(self):
        cleaned_data = super().clean()
        unit_price = cleaned_data.get('unit_price')
        quantity = cleaned_data.get('quantity')
        discount = cleaned_data.get('discount')
        
        # Calculate total price
        if unit_price is not None and quantity is not None:
            if discount is None:
                discount = 0
            
            if unit_price < 0:
                self.add_error('unit_price', 'Unit price cannot be negative')
            
            if quantity <= 0:
                self.add_error('quantity', 'Quantity must be positive')
            
            if discount < 0:
                self.add_error('discount', 'Discount cannot be negative')
        
        return cleaned_data

class PaymentForm(forms.ModelForm):
    class Meta:
        model = Payment
        fields = ['amount', 'payment_method', 'transaction_id', 'notes']
        widgets = {
            'notes': forms.Textarea(attrs={'rows': 2, 'class': 'form-control'}),
        }
    
    def __init__(self, *args, **kwargs):
        invoice = kwargs.pop('invoice', None)
        super().__init__(*args, **kwargs)
        
        if invoice:
            # Set default amount to the amount due
            self.fields['amount'].initial = invoice.amount_due
            
            # Validation for maximum payment
            self.invoice = invoice
    
    def clean_amount(self):
        amount = self.cleaned_data.get('amount')
        
        if hasattr(self, 'invoice') and amount > self.invoice.amount_due:
            raise forms.ValidationError(f"Payment amount cannot exceed the amount due (${self.invoice.amount_due})")
        
        if amount <= 0:
            raise forms.ValidationError("Payment amount must be positive")
        
        return amount
