from django import forms
from django.forms import inlineformset_factory
from .models import PharmacyLocation, MedicationInventory, PrescriptionDispensing, DispensedMedication, MedicationDispensing, PrescriptionInvoice, MedicationDelivery
from prescription.models import Medication

class PharmacyLocationForm(forms.ModelForm):
    class Meta:
        model = PharmacyLocation
        fields = ['name', 'address', 'phone', 'email', 'operating_hours', 'is_active']
        widgets = {
            'address': forms.Textarea(attrs={'rows': 3, 'class': 'form-control'}),
            'operating_hours': forms.TextInput(attrs={'class': 'form-control', 
                                                     'placeholder': 'Mon-Fri: 9AM-6PM, Sat: 10AM-4PM'}),
        }

class MedicationInventoryForm(forms.Form):
    """Form for managing medication inventory"""
    name = forms.CharField(max_length=100)
    dosage = forms.CharField(max_length=50)
    quantity = forms.IntegerField(min_value=0)
    expiration_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))
    notes = forms.CharField(widget=forms.Textarea(attrs={'rows': 3}), required=False)

class InventoryUpdateForm(forms.Form):
    """Form for updating inventory quantities"""
    quantity_change = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    reason = forms.CharField(widget=forms.Textarea(attrs={'rows': 2, 'class': 'form-control'}))
    
    def clean_quantity_change(self):
        quantity = self.cleaned_data.get('quantity_change')
        
        if quantity == 0:
            raise forms.ValidationError('Quantity change cannot be zero')
        
        return quantity

class DispensationForm(forms.ModelForm):
    """Form for medication dispensation"""
    class Meta:
        model = PrescriptionDispensing
        fields = ['notes']
        widgets = {
            'notes': forms.Textarea(attrs={'rows': 3})
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.update({'class': 'form-control'})


class DispensationItemForm(forms.ModelForm):
    """Form for dispensation items"""
    class Meta:
        model = DispensedMedication
        fields = ['prescription_item', 'inventory_item', 'quantity_dispensed']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.update({'class': 'form-control'})


# FormSet for tracking dispensed medications
DispensedMedicationFormSet = forms.inlineformset_factory(
    PrescriptionDispensing,
    MedicationDispensing,
    fields=['notes'],
    extra=1,
    can_delete=True
)

class MedicationForm(forms.ModelForm):
    """Form for adding/editing medications"""
    class Meta:
        model = Medication
        fields = ['name', 'description', 'dosage_form', 'strength', 'manufacturer']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.update({'class': 'form-control'})

class PrescriptionDispensingForm(forms.ModelForm):
    """Form for dispensing prescriptions"""
    class Meta:
        model = PrescriptionDispensing
        fields = ['status', 'notes']
        widgets = {
            'notes': forms.Textarea(attrs={'rows': 3}),
        }
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove 'pharmacy' field reference that doesn't exist in the model
        if 'pharmacy' in self.fields:
            del self.fields['pharmacy']

class MedicationDispensingForm(forms.ModelForm):
    """Form for individual medications being dispensed"""
    class Meta:
        model = MedicationDispensing
        fields = ['medication_name', 'dosage', 'quantity', 'notes']
        widgets = {
            'notes': forms.Textarea(attrs={'rows': 2}),
        }

# Create a formset for medication items
MedicationFormSet = forms.inlineformset_factory(
    PrescriptionDispensing,
    MedicationDispensing,
    form=MedicationDispensingForm,
    extra=1,
    can_delete=True
)

class PrescriptionInvoiceForm(forms.ModelForm):
    class Meta:
        model = PrescriptionInvoice
        fields = ['due_date', 'total_amount', 'notes']
        widgets = {
            'due_date': forms.DateInput(attrs={'type': 'date'}),
            'notes': forms.Textarea(attrs={'rows': 3}),
        }

class MedicationDeliveryForm(forms.ModelForm):
    class Meta:
        model = MedicationDelivery
        fields = ['delivery_address', 'tracking_number', 'estimated_delivery', 'status', 'delivery_notes']
        widgets = {
            'estimated_delivery': forms.DateTimeInput(attrs={'type': 'datetime-local'}),
            'delivery_notes': forms.Textarea(attrs={'rows': 3}),
        }
