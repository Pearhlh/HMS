from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.views.generic import ListView, DetailView, CreateView, UpdateView
from django.urls import reverse_lazy, reverse
from django.contrib import messages
from django.db.models import Q, Sum, F, ExpressionWrapper, DecimalField
from django.utils import timezone
from django.http import HttpResponse, HttpResponseForbidden
from django.db.utils import OperationalError
from django.db import transaction
import csv

from .models import PharmacyLocation, MedicationInventory, PrescriptionDispensing, DispensedMedication, MedicationDispensing, PrescriptionInvoice, MedicationDelivery
from .forms import MedicationInventoryForm, PrescriptionDispensingForm, DispensedMedicationFormSet, MedicationFormSet, PrescriptionInvoiceForm, MedicationDeliveryForm
from prescription.models import Prescription, Medication, PrescriptionItem
from notification.services import send_prescription_notification
from notification.models import NotificationRecord

class PharmacistRequiredMixin(UserPassesTestMixin):
    """Mixin to ensure user is a pharmacist"""
    def test_func(self):
        return self.request.user.is_pharmacist or self.request.user.is_admin


def is_pharmacy_staff(user):
    """Check if user is pharmacy staff or admin"""
    return user.is_pharmacist or user.is_admin or user.is_superuser


@login_required
def dashboard(request):
    """Pharmacy dashboard"""
    # Check all possible pharmacy role names
    if request.user.role not in ['pharmacy', 'pharmacist']: 
        return HttpResponseForbidden("You don't have access to this page.")
    
    # Get pending prescriptions
    try:
        pending_prescriptions = Prescription.objects.filter(status='pending').order_by('-date_prescribed')
    except Exception as e:
        print(f"Error loading prescriptions: {e}")
        pending_prescriptions = []
    
    context = {
        'pending_prescriptions': pending_prescriptions,
    }
    return render(request, 'pharmacy/dashboard.html', context)


@login_required
def prescription_list(request):
    """View all prescriptions"""
    # Check all possible pharmacy role names
    if request.user.role not in ['pharmacy', 'pharmacist']:
        return HttpResponseForbidden("You don't have access to this page.")
        
    try:
        prescriptions = Prescription.objects.all().order_by('-date_prescribed')
        
        # Filter by status if provided
        status = request.GET.get('status', '')
        if status:
            prescriptions = prescriptions.filter(status=status)
    except Exception as e:
        print(f"Error loading prescriptions: {e}")
        prescriptions = []
    
    context = {
        'prescriptions': prescriptions,
    }
    return render(request, 'pharmacy/prescription_list.html', context)


@login_required

def inventory_list(request):
    """View for listing medication inventory"""
    # Placeholder implementation
    return render(request, 'pharmacy/inventory_list.html', {})


@login_required
def add_inventory(request):
    """View for adding inventory"""
    # Placeholder implementation
    return render(request, 'pharmacy/add_inventory.html', {})


@login_required
def pharmacy_prescriptions(request):
    """View for listing prescriptions to be filled"""
    # Placeholder implementation
    return render(request, 'pharmacy/prescriptions.html', {})


@login_required
def dispense_prescription(request, pk):
    """View for dispensing medications for a prescription"""
    # Placeholder implementation
    return render(request, 'pharmacy/dispense.html', {})


@login_required
def process_prescription(request, prescription_id):
    """Process a prescription"""
    if request.user.role not in ['pharmacy', 'pharmacist']:
        return HttpResponseForbidden("You don't have access to this page.")
    
    prescription = get_object_or_404(Prescription, id=prescription_id)
    
    try:
        # Check if this prescription has already been processed
        try:
            dispensing = PrescriptionDispensing.objects.get(prescription=prescription)
        except PrescriptionDispensing.DoesNotExist:
            # Create a new dispensing object if one doesn't exist
            if request.method == 'POST':
                dispensing = None
            else:
                # For GET requests, initialize a new object but don't save it yet
                dispensing = PrescriptionDispensing(
                    prescription=prescription,
                    patient=prescription.patient,
                    pharmacist=request.user
                )
                # Don't save it yet - just for form initialization
        
        if request.method == 'POST':
            form = PrescriptionDispensingForm(request.POST, instance=dispensing)
            if form.is_valid():
                # Save the dispensing object
                dispensing = form.save(commit=False)
                if not dispensing.id:  # If it's a new dispensing
                    dispensing.prescription = prescription
                    dispensing.patient = prescription.patient
                    dispensing.pharmacist = request.user
                
                # Set dates based on status
                if dispensing.status == 'verified' and not dispensing.verified_date:
                    dispensing.verified_date = timezone.now()
                if dispensing.status == 'dispensed' and not dispensing.dispensed_date:
                    dispensing.dispensed_date = timezone.now()
                    
                dispensing.save()
                
                # Also update the prescription status
                prescription.status = dispensing.status
                prescription.save()
                
                messages.success(request, f"Prescription has been processed successfully.")
                return redirect('pharmacy:process_prescription', prescription_id=prescription.id)
        else:
            # Use the existing dispensing object or the newly created one for the form
            form = PrescriptionDispensingForm(instance=dispensing)
    
    except OperationalError:
        # Handle case where tables don't exist yet
        messages.warning(request, "The pharmacy system tables have not been created yet. Please run migrations first.")
        dispensing = None
        form = None
    
    context = {
        'prescription': prescription,
        'dispensing': dispensing,  # This will now be either a saved object or None
        'form': form,
        'table_missing': form is None,
    }
    return render(request, 'pharmacy/process_prescription.html', context)


@login_required
def create_invoice(request, dispensing_id):
    """Create an invoice for a prescription dispensing"""
    if request.user.role not in ['pharmacy', 'pharmacist']:
        return HttpResponseForbidden("You don't have access to this page.")
    
    try:
        dispensing = get_object_or_404(PrescriptionDispensing, id=dispensing_id)
        
        # Check if an invoice already exists
        try:
            invoice = PrescriptionInvoice.objects.get(prescription_dispensing=dispensing)
            messages.warning(request, "An invoice already exists for this prescription.")
            return redirect('pharmacy:view_invoice', invoice_id=invoice.id)
        except PrescriptionInvoice.DoesNotExist:
            pass
        
        if request.method == 'POST':
            form = PrescriptionInvoiceForm(request.POST)
            if form.is_valid():
                with transaction.atomic():
                    invoice = form.save(commit=False)
                    invoice.prescription_dispensing = dispensing
                    invoice.patient = dispensing.patient
                    invoice.pharmacist = request.user
                    invoice.save()
                    
                    # Notify the patient about the invoice
                    NotificationRecord.objects.create(
                        user=dispensing.patient.user,
                        subject='New Pharmacy Invoice Available',
                        message=f'An invoice for your prescription has been created. Total amount: ${invoice.total_amount}',
                        notification_type='invoice',
                        action_url=f'/patient/invoices/{invoice.id}/',  # Adjust this to match your actual URL pattern
                        action_text='View Invoice'
                    )
                    
                messages.success(request, f"Invoice #{invoice.invoice_number} created successfully.")
                return redirect('pharmacy:view_invoice', invoice_id=invoice.id)
        else:
            # Pre-calculate total based on medications
            total = 0
            try:
                for medication in dispensing.medications.all():
                    # In a real app, you'd have pricing data
                    # For now, use a placeholder calculation
                    total += medication.quantity * 10  # Assuming $10 per unit
            except:
                pass
                
            form = PrescriptionInvoiceForm(initial={
                'total_amount': total,
                'due_date': (timezone.now() + timezone.timedelta(days=30)).date()
            })
        
    except OperationalError:
        messages.warning(request, "The pharmacy system tables have not been created yet.")
        return redirect('pharmacy:dashboard')
    
    context = {
        'form': form,
        'dispensing': dispensing,
    }
    return render(request, 'pharmacy/create_invoice.html', context)


@login_required
def view_invoice(request, invoice_id):
    """View a prescription invoice"""
    if request.user.role not in ['pharmacy', 'pharmacist']:
        return HttpResponseForbidden("You don't have access to this page.")
    
    invoice = get_object_or_404(PrescriptionInvoice, id=invoice_id)
    
    # Calculate medication costs for display if needed
    medication_costs = []
    try:
        for medication in invoice.prescription_dispensing.medications.all():
            # Simple price calculation ($10 per unit)
            price_per_unit = 10.00
            item_total = medication.quantity * price_per_unit
            medication_costs.append({
                'medication': medication,
                'price_per_unit': price_per_unit,
                'item_total': item_total
            })
    except:
        # Handle case where medications can't be accessed
        pass
    
    context = {
        'invoice': invoice,
        'medication_costs': medication_costs
    }
    return render(request, 'pharmacy/view_invoice.html', context)


@login_required
def manage_delivery(request, dispensing_id):
    """Create or update delivery information and send confirmation"""
    if request.user.role not in ['pharmacy', 'pharmacist']:
        return HttpResponseForbidden("You don't have access to this page.")
    
    dispensing = get_object_or_404(PrescriptionDispensing, id=dispensing_id)
    
    # Get or create a delivery record
    try:
        delivery = MedicationDelivery.objects.get(prescription_dispensing=dispensing)
    except MedicationDelivery.DoesNotExist:
        delivery = MedicationDelivery(
            prescription_dispensing=dispensing,
            patient=dispensing.patient,
            delivery_address=dispensing.patient.address if hasattr(dispensing.patient, 'address') else ''
        )
    
    if request.method == 'POST':
        form = MedicationDeliveryForm(request.POST, instance=delivery)
        if form.is_valid():
            delivery = form.save(commit=False)
            
            # If status changed to delivered, set delivery date
            was_delivered = delivery.id and delivery.status == 'delivered'
            if not was_delivered and form.cleaned_data['status'] == 'delivered':
                delivery.actual_delivery = timezone.now()
            
            delivery.save()
            
            # Send notification if delivered and notification not yet sent
            if delivery.status == 'delivered' and not delivery.notification_sent:
                try:
                    # Debug info
                    print(f"Creating delivery notification for user {dispensing.patient.user.id}")
                    
                    notification = NotificationRecord.objects.create(
                        user=dispensing.patient.user,
                        subject="Medication Delivery Confirmation",  # Using subject field
                        message=f"Your medication has been delivered. Please check your delivery at: {delivery.delivery_address}",
                        notification_type="delivery",
                        action_url="/patient/prescriptions/",
                        action_text="View Prescriptions"
                    )
                    print(f"Created notification ID: {notification.id}")
                    
                    delivery.notification_sent = True
                    delivery.save(update_fields=['notification_sent'])
                    
                    messages.success(request, "Delivery marked as complete and notification sent.")
                except Exception as e:
                    print(f"Error creating notification: {str(e)}")
                    messages.warning(request, f"Delivery updated but notification failed: {str(e)}")
            else:
                messages.success(request, "Delivery information updated.")
                
            return redirect('pharmacy:view_delivery', delivery_id=delivery.id)
    else:
        form = MedicationDeliveryForm(instance=delivery)
    
    context = {
        'form': form,
        'dispensing': dispensing,
        'delivery': delivery,
    }
    return render(request, 'pharmacy/manage_delivery.html', context)


@login_required
def view_delivery(request, delivery_id):
    """View delivery details"""
    if request.user.role not in ['pharmacy', 'pharmacist']:
        return HttpResponseForbidden("You don't have access to this page.")
    
    delivery = get_object_or_404(MedicationDelivery, id=delivery_id)
    
    context = {
        'delivery': delivery,
    }
    return render(request, 'pharmacy/view_delivery.html', context)
