from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponseForbidden
from django.utils import timezone
from django.db.models import Q, Count

from .models import Doctor
from .forms import DoctorForm, DoctorUserForm
from appointment.models import Appointment
from patient.models import Patient, MedicalRecord
from prescription.models import Prescription

@login_required
def dashboard(request):
    """Doctor dashboard view"""
    if not request.user.role == 'doctor':
        return HttpResponseForbidden("You don't have access to this page.")
    
    try:
        doctor = Doctor.objects.get(user=request.user)
    except Doctor.DoesNotExist:
        doctor = Doctor.objects.create(user=request.user)
    
    today = timezone.now().date()
    
    today_appointments = Appointment.objects.filter(
        doctor=doctor,
        appointment_date=today
    ).order_by('appointment_time')
    
    upcoming_appointments = Appointment.objects.filter(
        doctor=doctor,
        appointment_date__gt=today,
        status__in=['scheduled', 'confirmed']
    ).order_by('appointment_date', 'appointment_time')[:10]
    
    pending_appointments = Appointment.objects.filter(
        doctor=doctor,
        status='scheduled'
    ).order_by('appointment_date', 'appointment_time')
    
    recent_patients = Patient.objects.filter(
        appointments__doctor=doctor
    ).distinct()[:10]
    
    context = {
        'doctor': doctor,
        'today_appointments': today_appointments,
        'upcoming_appointments': upcoming_appointments,
        'pending_appointments': pending_appointments,
        'today_appointments_count': today_appointments.count(),
        'upcoming_appointments_count': upcoming_appointments.count(),
        'pending_appointments_count': pending_appointments.count(),
        'recent_patients': recent_patients
    }
    return render(request, 'doctor/dashboard.html', context)

@login_required
def appointment_list(request):
    """View all doctor appointments"""
    if not request.user.role == 'doctor':
        return HttpResponseForbidden("You don't have access to this page.")
    
    try:
        doctor = Doctor.objects.get(user=request.user)
    except Doctor.DoesNotExist:
        doctor = Doctor.objects.create(user=request.user)
    
    status = request.GET.get('status', '')
    date_from = request.GET.get('date_from', '')
    date_to = request.GET.get('date_to', '')
    
    appointments = Appointment.objects.filter(doctor=doctor).order_by('appointment_date', 'appointment_time')
    
    if status:
        appointments = appointments.filter(status=status)
    
    if date_from:
        appointments = appointments.filter(appointment_date__gte=date_from)
    
    if date_to:
        appointments = appointments.filter(appointment_date__lte=date_to)
    
    context = {
        'appointments': appointments,
        'doctor': doctor,
        'today': timezone.now().date()  # Add today's date for template comparison
    }
    return render(request, 'doctor/appointments/list.html', context)

@login_required
def appointment_manage(request, pk):
    """Accept or decline appointment"""
    if not request.user.role == 'doctor':
        return HttpResponseForbidden("You don't have access to this page.")
        
    appointment = get_object_or_404(Appointment, pk=pk, doctor__user=request.user)
    
    if request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'accept':
            appointment.status = 'confirmed'
            appointment.save()
            messages.success(request, f"Appointment with {appointment.patient.user.get_full_name()} on {appointment.appointment_date} has been confirmed.")
        elif action == 'decline':
            appointment.status = 'canceled'
            appointment.save()
            messages.success(request, f"Appointment with {appointment.patient.user.get_full_name()} has been declined.")
            
    return redirect('doctor:dashboard')

@login_required
def patient_list(request):
    """View all patients for this doctor"""
    if not request.user.role == 'doctor':
        return HttpResponseForbidden("You don't have access to this page.")
    
    try:
        doctor = Doctor.objects.get(user=request.user)
    except Doctor.DoesNotExist:
        doctor = Doctor.objects.create(user=request.user)
    
    patients = Patient.objects.filter(
        appointments__doctor=doctor
    ).distinct()
    
    context = {
        'patients': patients,
        'doctor': doctor,
    }
    return render(request, 'doctor/patients/list.html', context)

@login_required
def patient_detail(request, pk):
    """View and update patient details"""
    if not request.user.role == 'doctor':
        return HttpResponseForbidden("You don't have access to this page.")
        
    patient = get_object_or_404(Patient, pk=pk)
    
    # Check if this doctor has seen this patient
    doctor = Doctor.objects.get(user=request.user)
    has_appointment = Appointment.objects.filter(doctor=doctor, patient=patient).exists()
    
    if not has_appointment and not request.user.is_staff:
        return HttpResponseForbidden("You don't have access to this patient's records.")
    
    # Get medical records for this patient
    medical_records = MedicalRecord.objects.filter(patient=patient).order_by('-record_date')
    
    # Get appointments with this patient
    appointments = Appointment.objects.filter(
        doctor=doctor, 
        patient=patient
    ).order_by('-appointment_date')
    
    # Get prescriptions for this patient
    prescriptions = Prescription.objects.filter(
        patient=patient
    ).order_by('-date_prescribed')
    
    # Get lab tests with error handling for missing columns
    try:
        from .utils import get_lab_tests_safely
        lab_tests = get_lab_tests_safely(pk)
    except Exception as e:
        print(f"Error retrieving lab tests: {e}")
        lab_tests = []
        messages.warning(request, "Some lab test data may not be available due to system updates.")
    
    # Get vital signs recorded by nurses
    try:
        from nurse.models import VitalSigns
        vitals_history = VitalSigns.objects.filter(patient=patient).order_by('-recorded_at')[:5]
    except (ImportError, Exception) as e:
        print(f"Error retrieving vital signs: {str(e)}")
        vitals_history = []
    
    # Get vital sign notifications
    try:
        from notification.models import NotificationRecord
        vital_notifications = NotificationRecord.objects.filter(
            user=request.user,
            notification_type="care_note", 
            action_url__contains=f"/doctor/patient/{patient.id}/"
        ).order_by('-created_at')[:5]
    except Exception as e:
        print(f"Error retrieving notifications: {str(e)}")
        vital_notifications = []
    
    context = {
        'patient': patient,
        'medical_records': medical_records,
        'appointments': appointments,
        'prescriptions': prescriptions,
        'lab_tests': lab_tests,
        'vitals_history': vitals_history,
        'vital_notifications': vital_notifications,
        'now': timezone.now(),  # Add current date/time for display in the template
    }
    return render(request, 'doctor/patients/detail.html', context)

@login_required
def medical_record_add(request, patient_id):
    """Add new medical record for a patient"""
    if not request.user.role == 'doctor':
        return HttpResponseForbidden("You don't have access to this page.")
        
    patient = get_object_or_404(Patient, pk=patient_id)
    doctor = Doctor.objects.get(user=request.user)
    
    if request.method == 'POST':
        # Create new medical record with the fields that exist in the model
        # Remove 'doctor' field if it doesn't exist in the MedicalRecord model
        record = MedicalRecord.objects.create(
            patient=patient,
            record_date=timezone.now().date(),
            diagnosis=request.POST.get('diagnosis'),
            treatment=request.POST.get('treatment'),
            notes=request.POST.get('notes'),
            created_by=request.user
        )
        
        messages.success(request, "Medical record added successfully")
        return redirect('doctor:patient_detail', pk=patient_id)
        
    context = {
        'patient': patient
    }
    return render(request, 'doctor/medical_records/add.html', context)

@login_required
def prescription_add(request, patient_id):
    """Create new prescription for a patient"""
    if not request.user.role == 'doctor':
        return HttpResponseForbidden("You don't have access to this page.")
        
    patient = get_object_or_404(Patient, pk=patient_id)
    doctor = Doctor.objects.get(user=request.user)
    
    if request.method == 'POST':
        prescription = Prescription.objects.create(
            patient=patient,
            doctor=doctor,
            date_prescribed=timezone.now().date(),
            notes=request.POST.get('notes'),
            status='pending'
        )
        
        medication = request.POST.get('medication')
        dosage = request.POST.get('dosage')
        frequency = request.POST.get('frequency')
        duration = request.POST.get('duration')
        
        prescription.notes = f"Medication: {medication}\nDosage: {dosage}\nFrequency: {frequency}\nDuration: {duration}\n\n{prescription.notes}"
        prescription.save()
        
        messages.success(request, "Prescription created and sent to pharmacy")
        return redirect('doctor:patient_detail', pk=patient_id)
        
    context = {
        'patient': patient
    }
    return render(request, 'doctor/prescriptions/add.html', context)

@login_required
def request_lab_test(request, patient_id):
    """Submit a new lab test request"""
    if not request.user.role == 'doctor':
        return HttpResponseForbidden("You don't have access to this page.")
        
    patient = get_object_or_404(Patient, pk=patient_id)
    doctor = get_object_or_404(Doctor, user=request.user)
    
    if request.method == 'POST':
        test_type = request.POST.get('test_type')
        description = request.POST.get('description')
        instructions = request.POST.get('instructions', '')
        
        try:
            # Add debug message to verify data
            print(f"Creating lab test: {test_type} for patient {patient.id} by doctor {doctor.id}")
            
            # Import the LabTest model
            from laboratory.models import LabTest
            
            # Create the lab test record in the database
            lab_test = LabTest.objects.create(
                patient=patient,
                requested_by=doctor,
                test_type=test_type,
                description=description,
                instructions=instructions,
                status='requested'
            )
            
            # Print confirmation of successful creation
            print(f"Lab test created with ID: {lab_test.id}")
            
            messages.success(request, f"Test request for {lab_test.get_test_type_display()} has been sent to the laboratory.")
            return redirect('doctor:patient_detail', pk=patient_id)
        except ImportError:
            messages.error(request, "Laboratory module is not available. Please contact the administrator.")
            return redirect('doctor:patient_detail', pk=patient_id)
        except Exception as e:
            # Print the detailed error for debugging
            import traceback
            print(traceback.format_exc())
            messages.error(request, f"Error creating lab test request: {str(e)}")
    
    context = {
        'patient': patient,
    }
    return render(request, 'doctor/lab_tests/request_test.html', context)

@login_required
def lab_test_detail(request, patient_id, test_id):
    """View lab test details"""
    if not request.user.role == 'doctor':
        return HttpResponseForbidden("You don't have access to this page.")
        
    patient = get_object_or_404(Patient, pk=patient_id)
    
    # Get actual lab test from database if available
    try:
        from laboratory.models import LabTest
        test = get_object_or_404(LabTest, pk=test_id, patient=patient)
    except (ImportError, Exception) as e:
        # Create a dictionary with the same structure as the LabTest model
        print(f"Error retrieving lab test: {str(e)}")
        test = {
            'id': test_id,
            'test_type': 'Blood Test',
            'get_test_type_display': lambda: 'Blood Test',
            'requested_date': timezone.now(),
            'status': 'pending',
            'get_status_display': lambda: 'Pending',
            'description': 'Complete blood count and metabolic panel',
            'instructions': 'Patient should fast for 12 hours before the test',
            'results': None
        }
    
    context = {
        'patient': patient,
        'test': test,
    }
    return render(request, 'doctor/lab_tests/detail.html', context)
