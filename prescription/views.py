from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.utils import timezone
from django.db.models import Q
from .models import Prescription, PrescriptionItem, Medication
from .forms import PrescriptionForm, PrescriptionItemFormset, MedicationForm

def is_medical_staff(user):
    """Check if user is medical staff (doctor, nurse, or pharmacist)"""
    return user.is_doctor or user.is_nurse or user.is_pharmacist or user.is_admin or user.is_superuser

@login_required
def prescription_list(request):
    """View for listing prescriptions"""
    user = request.user
    
    # Filter prescriptions based on user role
    if user.is_patient:
        try:
            patient = user.patient_profile
            prescriptions = Prescription.objects.filter(patient=patient)
        except:
            prescriptions = Prescription.objects.none()
    elif user.is_doctor:
        try:
            doctor = user.doctor_profile
            prescriptions = Prescription.objects.filter(doctor=doctor)
        except:
            prescriptions = Prescription.objects.none()
    elif user.is_pharmacist or user.is_admin or user.is_superuser:
        prescriptions = Prescription.objects.all()
    else:
        prescriptions = Prescription.objects.none()
        messages.error(request, "You don't have permission to view prescriptions.")
    
    # Apply filters
    status = request.GET.get('status')
    if status:
        prescriptions = prescriptions.filter(status=status)
    
    search = request.GET.get('search')
    if search:
        if user.is_patient:
            # Patients can only search by doctor name
            prescriptions = prescriptions.filter(
                Q(doctor__user__first_name__icontains=search) |
                Q(doctor__user__last_name__icontains=search)
            )
        else:
            # Medical staff can search by patient or doctor
            prescriptions = prescriptions.filter(
                Q(patient__user__first_name__icontains=search) |
                Q(patient__user__last_name__icontains=search) |
                Q(doctor__user__first_name__icontains=search) |
                Q(doctor__user__last_name__icontains=search)
            )
    
    # Sorting
    sort = request.GET.get('sort', '-date')
    if sort == 'date':
        prescriptions = prescriptions.order_by('date_prescribed')
    elif sort == '-date':
        prescriptions = prescriptions.order_by('-date_prescribed')
    
    context = {
        'prescriptions': prescriptions,
        'status_filter': status,
        'search': search,
        'sort': sort,
    }
    
    return render(request, 'prescription/prescription_list.html', context)

@login_required
def prescription_detail(request, pk):
    """View for prescription details"""
    prescription = get_object_or_404(Prescription, pk=pk)
    
    # Check if user has permission to view this prescription
    user = request.user
    if user.is_patient and prescription.patient.user != user:
        messages.error(request, "You don't have permission to view this prescription")
        return redirect('prescription_list')
    
    context = {
        'prescription': prescription,
        'items': prescription.items.all()
    }
    
    return render(request, 'prescription/prescription_detail.html', context)

@login_required
@user_passes_test(is_medical_staff)
def create_prescription(request):
    """View for creating a new prescription"""
    if request.method == 'POST':
        form = PrescriptionForm(request.POST, user=request.user)
        
        if form.is_valid():
            prescription = form.save(commit=False)
            
            # If user is a doctor, automatically set the doctor
            if request.user.is_doctor:
                prescription.doctor = request.user.doctor_profile
            
            prescription.save()
            
            # Process the formset
            formset = PrescriptionItemFormset(request.POST, instance=prescription)
            if formset.is_valid():
                formset.save()
                messages.success(request, "Prescription created successfully")
                
                # Send notification
                from notification.services import send_prescription_notification
                send_prescription_notification(prescription)
                
                return redirect('prescription_detail', pk=prescription.id)
            else:
                # If formset is invalid, delete the prescription to avoid orphaned records
                prescription.delete()
    else:
        # Pre-select patient if provided in query params
        patient_id = request.GET.get('patient')
        initial = {}
        if patient_id:
            try:
                from patient.models import Patient
                patient = Patient.objects.get(pk=patient_id)
                initial['patient'] = patient
            except Patient.DoesNotExist:
                pass
        
        form = PrescriptionForm(user=request.user, initial=initial)
        formset = PrescriptionItemFormset()
    
    medications = Medication.objects.all()
    
    context = {
        'form': form,
        'formset': formset,
        'medications': medications
    }
    
    return render(request, 'prescription/create_prescription.html', context)

@login_required
@user_passes_test(is_medical_staff)
def update_prescription(request, pk):
    """View for updating a prescription"""
    prescription = get_object_or_404(Prescription, pk=pk)
    
    # Only allow updates for pending prescriptions
    if prescription.status != 'pending':
        messages.error(request, "Only pending prescriptions can be updated")
        return redirect('prescription_detail', pk=prescription.id)
    
    # Only the prescribing doctor or admin can update
    if request.user.is_doctor and prescription.doctor.user != request.user and not request.user.is_admin:
        messages.error(request, "You don't have permission to update this prescription")
        return redirect('prescription_list')
    
    if request.method == 'POST':
        form = PrescriptionForm(request.POST, instance=prescription, user=request.user)
        formset = PrescriptionItemFormset(request.POST, instance=prescription)
        
        if form.is_valid() and formset.is_valid():
            form.save()
            formset.save()
            messages.success(request, "Prescription updated successfully")
            return redirect('prescription_detail', pk=prescription.id)
    else:
        form = PrescriptionForm(instance=prescription, user=request.user)
        formset = PrescriptionItemFormset(instance=prescription)
    
    context = {
        'form': form,
        'formset': formset,
        'prescription': prescription
    }
    
    return render(request, 'prescription/update_prescription.html', context)

@login_required
@user_passes_test(lambda u: u.is_pharmacist or u.is_admin or u.is_superuser)
def verify_prescription(request, pk):
    """View for pharmacist to verify a prescription"""
    prescription = get_object_or_404(Prescription, pk=pk)
    
    # Only verify pending prescriptions
    if prescription.status != 'pending':
        messages.error(request, "This prescription has already been processed")
        return redirect('prescription_detail', pk=prescription.id)
    
    if request.method == 'POST':
        prescription.status = 'verified'
        prescription.save()
        
        messages.success(request, "Prescription verified successfully")
        
        # Send notification
        from notification.services import send_prescription_notification
        send_prescription_notification(prescription)
        
        return redirect('prescription_detail', pk=prescription.id)
    
    return render(request, 'prescription/verify_prescription.html', {'prescription': prescription})

@login_required
@user_passes_test(lambda u: u.is_pharmacist or u.is_admin or u.is_superuser)
def dispense_prescription(request, pk):
    """View for pharmacist to dispense medications"""
    prescription = get_object_or_404(Prescription, pk=pk)
    
    # Only verified prescriptions can be dispensed
    if prescription.status != 'verified':
        if prescription.status == 'dispensed':
            messages.error(request, "This prescription has already been dispensed")
        else:
            messages.error(request, "Only verified prescriptions can be dispensed")
        return redirect('prescription_detail', pk=prescription.id)
    
    if request.method == 'POST':
        # In a real system, we would check inventory and decrement stock here
        prescription.status = 'dispensed'
        prescription.save()
        
        messages.success(request, "Prescription dispensed successfully")
        
        # Send notification
        from notification.services import send_prescription_notification
        send_prescription_notification(prescription)
        
        # Redirect to a "Print Label" page in a real system
        return redirect('prescription_detail', pk=prescription.id)
    
    context = {
        'prescription': prescription,
        'items': prescription.items.all()
    }
    
    return render(request, 'prescription/dispense_prescription.html', context)

@login_required
@user_passes_test(is_medical_staff)
def medication_list(request):
    """View for listing medications"""
    medications = Medication.objects.all()
    
    # Apply search filter
    search = request.GET.get('search')
    if search:
        medications = medications.filter(
            Q(name__icontains=search) |
            Q(description__icontains=search) |
            Q(dosage_form__icontains=search) |
            Q(strength__icontains=search) |
            Q(manufacturer__icontains=search)
        )
    
    context = {
        'medications': medications,
        'search': search
    }
    
    return render(request, 'prescription/medication_list.html', context)

@login_required
@user_passes_test(lambda u: u.is_pharmacist or u.is_admin or u.is_superuser)
def add_medication(request):
    """View for adding a new medication"""
    if request.method == 'POST':
        form = MedicationForm(request.POST)
        if form.is_valid():
            medication = form.save()
            messages.success(request, f"Medication '{medication.name}' added successfully")
            return redirect('medication_list')
    else:
        form = MedicationForm()
    
    return render(request, 'prescription/add_medication.html', {'form': form})
