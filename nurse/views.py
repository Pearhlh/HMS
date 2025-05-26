from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import HttpResponseForbidden
from django.contrib import messages
from django.utils import timezone
from django.db import models
from .models import Nurse, NurseAssignment, VitalSigns
from patient.models import Patient
from .forms import VitalSignsForm
from notification.models import NotificationRecord

def is_nurse_or_admin(user):
    """Check if user is a nurse, admin, or superuser"""
    return user.role == 'nurse' or user.is_admin or user.is_superuser

@login_required
def dashboard(request):
    """Nurse dashboard view"""
    if not request.user.role == 'nurse':
        return HttpResponseForbidden("You do not have access to this page.")
    
    # Get recent patients whose vitals were updated
    recent_patients = Patient.objects.filter(
        vitals__recorded_by=request.user
    ).distinct().order_by('-vitals__recorded_at')[:5]
    
    context = {
        'user': request.user,
        'recent_patients': recent_patients
    }
    return render(request, 'nurse/dashboard.html', context)

@login_required
@user_passes_test(is_nurse_or_admin)
def nurse_dashboard(request):
    """Nurse dashboard view with assignments"""
    try:
        nurse = request.user.nurse_profile
        
        # Get nurse's current assignments
        active_assignments = NurseAssignment.objects.filter(
            nurse=request.user,
            is_active=True
        )
        
        # Get patients assigned to this nurse
        assigned_patients = Patient.objects.filter(
            nurse_assignments__nurse=request.user,
            nurse_assignments__is_active=True
        ).distinct()
        
        # Get recent vital signs recorded by this nurse
        recent_vitals = VitalSigns.objects.filter(
            recorded_by=request.user
        ).order_by('-recorded_at')[:10]
        
        context = {
            'nurse': nurse,
            'active_assignments': active_assignments,
            'assigned_patients': assigned_patients,
            'recent_vitals': recent_vitals,
        }
        
        return render(request, 'nurse/dashboard.html', context)
    
    except Nurse.DoesNotExist:
        messages.error(request, "Nurse profile not found. Please contact an administrator.")
        return redirect('home')

@login_required
@user_passes_test(is_nurse_or_admin)
def record_vitals(request, patient_id):
    """Record vital signs for a patient"""
    # Implementation would go here
    return render(request, 'nurse/record_vitals.html', {})

@login_required
def patient_list(request):
    """View all patients"""
    if not request.user.role == 'nurse':
        return HttpResponseForbidden("You do not have access to this page.")
    
    # Add search functionality
    search_query = request.GET.get('search', '')
    if search_query:
        patients = Patient.objects.filter(
            models.Q(user__first_name__icontains=search_query) | 
            models.Q(user__last_name__icontains=search_query) |
            models.Q(user__email__icontains=search_query)
        ).order_by('user__last_name', 'user__first_name')
    else:
        patients = Patient.objects.all().order_by('user__last_name', 'user__first_name')
    
    context = {
        'patients': patients,
        'search_query': search_query
    }
    return render(request, 'nurse/patient_list.html', context)

@login_required
def patient_detail(request, patient_id):
    """View patient details"""
    if not request.user.role == 'nurse':
        return HttpResponseForbidden("You do not have access to this page.")
    
    patient = get_object_or_404(Patient, id=patient_id)
    vitals_history = VitalSigns.objects.filter(patient=patient).order_by('-recorded_at')[:10]
    
    context = {
        'patient': patient,
        'vitals_history': vitals_history
    }
    return render(request, 'nurse/patient_detail.html', context)

@login_required
def update_vitals(request, patient_id):
    """Update vitals for a patient"""
    if not request.user.role == 'nurse':
        return HttpResponseForbidden("You do not have access to this page.")
    
    patient = get_object_or_404(Patient, id=patient_id)
    
    if request.method == 'POST':
        form = VitalSignsForm(request.POST)
        if form.is_valid():
            vitals = form.save(commit=False)
            vitals.patient = patient
            vitals.recorded_by = request.user
            vitals.recorded_at = timezone.now()
            vitals.save()
            
            # Create care notes and notifications
            care_note = request.POST.get('care_note')
            if care_note:
                # Create notification content
                subject = f"Care Note from {request.user.get_full_name()}"
                message = f"Care Note: {care_note}\n\n"
                message += f"Vitals recorded: Temperature: {vitals.temperature}°C, "
                message += f"Blood Pressure: {vitals.blood_pressure_systolic}/{vitals.blood_pressure_diastolic} mmHg, "
                message += f"Heart Rate: {vitals.heart_rate} bpm"
                
                # Send notification to patient
                try:
                    NotificationRecord.objects.create(
                        user=patient.user,
                        subject=subject,
                        message=message,
                        notification_type="care_note",
                        created_at=timezone.now(),  # Add this line
                        read=False,  # Add this line
                        action_url=f"/patient/vitals/{vitals.id}/",
                        action_text="View Vital Signs"
                    )
                except Exception as e:
                    messages.warning(request, f"Could not send notification to patient: {str(e)}")
                
                # Send notification to doctor if assigned
                try:
                    if hasattr(patient, 'doctor') and patient.doctor and patient.doctor.user:
                        NotificationRecord.objects.create(
                            user=patient.doctor.user,
                            subject=f"Care Note for Patient {patient.user.get_full_name()}",
                            message=message,
                            notification_type="care_note",
                            action_url=f"/doctor/patient/{patient.id}/",
                            action_text="View Patient"
                        )
                except Exception as e:
                    messages.warning(request, f"Could not send notification to doctor: {str(e)}")
                
                messages.success(request, "Vitals updated and care note sent to patient and doctor.")
            else:
                messages.success(request, f"Vitals updated for {patient.user.get_full_name()}")
                
            return redirect('nurse:patient_detail', patient_id=patient.id)
    else:
        form = VitalSignsForm()
    
    context = {
        'form': form,
        'patient': patient
    }
    return render(request, 'nurse/update_vitals.html', context)
