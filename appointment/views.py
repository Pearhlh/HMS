from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from django.db.models import Q
from .models import Appointment
from .forms import AppointmentForm, AppointmentStatusForm
from doctor.models import Doctor
from patient.models import Patient

@login_required
def appointment_list(request):
    """View for listing appointments"""
    user = request.user
    today = timezone.now().date()
    
    # Filter appointments based on user role
    if user.is_patient:
        try:
            patient = user.patient_profile
            appointments = Appointment.objects.filter(patient=patient)
        except:
            appointments = Appointment.objects.none()
            messages.error(request, "Patient profile not found.")
    elif user.is_doctor:
        try:
            doctor = user.doctor_profile
            appointments = Appointment.objects.filter(doctor=doctor)
        except:
            appointments = Appointment.objects.none()
            messages.error(request, "Doctor profile not found.")
    elif user.is_admin or user.is_staff:
        appointments = Appointment.objects.all()
    else:
        appointments = Appointment.objects.none()
        messages.error(request, "You don't have permission to view appointments.")
    
    # Apply filters if provided
    status = request.GET.get('status')
    if status:
        appointments = appointments.filter(status=status)
    
    date_filter = request.GET.get('date')
    if date_filter == 'today':
        appointments = appointments.filter(appointment_date=today)
    elif date_filter == 'upcoming':
        appointments = appointments.filter(appointment_date__gte=today)
    elif date_filter == 'past':
        appointments = appointments.filter(appointment_date__lt=today)
    
    search = request.GET.get('search')
    if search:
        if user.is_patient:
            # Patients can only search doctor names
            appointments = appointments.filter(
                Q(doctor__user__first_name__icontains=search) |
                Q(doctor__user__last_name__icontains=search)
            )
        else:
            # Staff can search patient and doctor names
            appointments = appointments.filter(
                Q(patient__user__first_name__icontains=search) |
                Q(patient__user__last_name__icontains=search) |
                Q(doctor__user__first_name__icontains=search) |
                Q(doctor__user__last_name__icontains=search)
            )
    
    # Sort appointments
    sort = request.GET.get('sort', 'date')
    if sort == 'date':
        appointments = appointments.order_by('appointment_date', 'appointment_time')
    elif sort == 'date_desc':
        appointments = appointments.order_by('-appointment_date', '-appointment_time')
    
    context = {
        'appointments': appointments,
        'status_filter': status,
        'date_filter': date_filter,
        'search': search,
        'sort': sort,
    }
    
    return render(request, 'appointment/appointment_list.html', context)

@login_required
def book_appointment(request):
    """View for booking a new appointment"""
    if request.method == 'POST':
        form = AppointmentForm(request.POST, user=request.user)
        if form.is_valid():
            appointment = form.save(commit=False)
            
            # If user is a patient, automatically set the patient
            if request.user.is_patient:
                appointment.patient = request.user.patient_profile
            
            appointment.status = 'scheduled'
            appointment.created_at = timezone.now()
            appointment.save()
            
            messages.success(request, "Appointment booked successfully")
            
            # Send notification
            from notification.services import send_appointment_notification
            send_appointment_notification(appointment)
            
            return redirect('appointment_detail', pk=appointment.id)
    else:
        # Pre-select doctor if provided in query params
        doctor_id = request.GET.get('doctor')
        initial = {}
        if doctor_id:
            try:
                doctor = Doctor.objects.get(pk=doctor_id)
                initial['doctor'] = doctor
            except Doctor.DoesNotExist:
                pass
        
        form = AppointmentForm(user=request.user, initial=initial)
    
    # Get available doctors for the form
    doctors = Doctor.objects.filter(user__is_active=True)
    
    context = {
        'form': form,
        'doctors': doctors
    }
    
    return render(request, 'appointment/book_appointment.html', context)

@login_required
def appointment_detail(request, pk):
    """View for appointment details"""
    appointment = get_object_or_404(Appointment, pk=pk)
    
    # Check if user has permission to view this appointment
    user = request.user
    if user.is_patient and appointment.patient.user != user:
        messages.error(request, "You don't have permission to view this appointment")
        return redirect('appointment_list')
    elif user.is_doctor and appointment.doctor.user != user:
        messages.error(request, "You don't have permission to view this appointment")
        return redirect('appointment_list')
    
    # Form for updating appointment status
    status_form = AppointmentStatusForm(initial={
        'status': appointment.status,
        'notes': appointment.notes
    })
    
    context = {
        'appointment': appointment,
        'status_form': status_form
    }
    
    return render(request, 'appointment/appointment_detail.html', context)

@login_required
def update_appointment(request, pk):
    """View for updating appointment details"""
    appointment = get_object_or_404(Appointment, pk=pk)
    
    # Check permissions
    user = request.user
    if user.is_patient and appointment.patient.user != user:
        messages.error(request, "You don't have permission to update this appointment")
        return redirect('appointment_list')
    elif user.is_doctor and appointment.doctor.user != user:
        messages.error(request, "You don't have permission to update this appointment")
        return redirect('appointment_list')
    
    if request.method == 'POST':
        if 'update_status' in request.POST:
            status_form = AppointmentStatusForm(request.POST)
            if status_form.is_valid():
                appointment.status = status_form.cleaned_data['status']
                appointment.notes = status_form.cleaned_data['notes']
                appointment.updated_at = timezone.now()
                appointment.save()
                
                messages.success(request, "Appointment status updated successfully")
                
                # Send notification
                from notification.services import send_appointment_notification
                send_appointment_notification(appointment)
                
                return redirect('appointment_detail', pk=appointment.id)
        else:
            form = AppointmentForm(request.POST, instance=appointment, user=user)
            if form.is_valid():
                form.save()
                messages.success(request, "Appointment updated successfully")
                return redirect('appointment_detail', pk=appointment.id)
    else:
        form = AppointmentForm(instance=appointment, user=user)
    
    context = {
        'form': form,
        'appointment': appointment
    }
    
    return render(request, 'appointment/update_appointment.html', context)

@login_required
def cancel_appointment(request, pk):
    """View for canceling an appointment"""
    appointment = get_object_or_404(Appointment, pk=pk)
    
    # Check permissions
    user = request.user
    if user.is_patient and appointment.patient.user != user:
        messages.error(request, "You don't have permission to cancel this appointment")
        return redirect('appointment_list')
    
    if request.method == 'POST':
        appointment.status = 'canceled'
        appointment.updated_at = timezone.now()
        appointment.save()
        
        messages.success(request, "Appointment canceled successfully")
        
        # Send notification
        from notification.services import send_appointment_notification
        send_appointment_notification(appointment)
        
        return redirect('appointment_list')
    
    return render(request, 'appointment/cancel_appointment.html', {'appointment': appointment})

@login_required
def confirm_appointment(request, pk):
    """View for confirming an appointment (primarily for staff)"""
    appointment = get_object_or_404(Appointment, pk=pk)
    
    # Only doctors and admins can confirm appointments
    if not (request.user.is_doctor or request.user.is_admin or request.user.is_staff):
        messages.error(request, "You don't have permission to confirm appointments")
        return redirect('appointment_list')
    
    if request.method == 'POST':
        appointment.status = 'confirmed'
        appointment.updated_at = timezone.now()
        appointment.save()
        
        messages.success(request, "Appointment confirmed successfully")
        
        # Send notification
        from notification.services import send_appointment_notification
        send_appointment_notification(appointment)
        
        return redirect('appointment_detail', pk=appointment.id)
    
    return render(request, 'appointment/confirm_appointment.html', {'appointment': appointment})

@login_required
def appointment_calendar(request):
    """Calendar view of appointments"""
    user = request.user
    today = timezone.now().date()
    
    if user.is_patient:
        try:
            patient = user.patient_profile
            appointments = Appointment.objects.filter(patient=patient)
        except:
            appointments = Appointment.objects.none()
    elif user.is_doctor:
        try:
            doctor = user.doctor_profile
            appointments = Appointment.objects.filter(doctor=doctor)
        except:
            appointments = Appointment.objects.none()
    elif user.is_admin or user.is_staff:
        appointments = Appointment.objects.all()
    else:
        appointments = Appointment.objects.none()
    
    # Filter for upcoming appointments
    upcoming_appointments = appointments.filter(appointment_date__gte=today)
    
    context = {
        'appointments': upcoming_appointments,
    }
    
    return render(request, 'appointment/calendar.html', context)
