from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.views.generic import ListView, DetailView, CreateView, UpdateView
from django.urls import reverse_lazy
from django.contrib import messages
from django.db.models import Q, Sum
import numpy as np
from django.utils import timezone
from django.http import HttpResponseForbidden, JsonResponse
from django.db.utils import OperationalError  
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from .models import Patient, MedicalRecord
from .forms import PatientForm, PatientUserForm, MedicalRecordForm
from appointment.models import Appointment, Department, Doctor
from doctor.models import Doctor as DoctorModel
from prescription.models import Prescription
from insurance.models import InsuranceClaim
from billing.models import Bill, Payment, PaymentMethod
from notification.models import NotificationRecord
from chatbot.diagnose import HealthDiagnosisBot
import logging
from datetime import timedelta, date

logger = logging.getLogger(__name__)
class PatientAccessMixin(UserPassesTestMixin):
    """Mixin to check if user is either the patient or a medical staff"""
    def test_func(self):
        patient = self.get_object()
        user = self.request.user
        
        # Patient can view their own records
        if hasattr(user, 'patient_profile') and user.patient_profile == patient:
            return True
            
        # Medical staff (doctors, nurses, admins) can access patient records
        return user.is_doctor or user.is_nurse or user.is_admin or user.is_superuser


@login_required
def dashboard(request):
    """Patient dashboard view"""
    # Check if user is a patient
    if not request.user.role == 'patient':
        return HttpResponseForbidden("You don't have access to this page.")
    
    # Get the patient object for the current user
    try:
        patient = Patient.objects.get(user=request.user)
    except Patient.DoesNotExist:
        # Create patient profile if it doesn't exist
        patient = Patient.objects.create(user=request.user)
    
    # Get relevant data using actual database queries
    today = timezone.now().date()
    
    # Get upcoming appointments
    try:
        upcoming_appointments = Appointment.objects.filter(
            patient=patient,
            appointment_date__gte=today,
            status__in=['scheduled', 'confirmed']
        ).order_by('appointment_date', 'appointment_time')[:5]
        upcoming_appointments_count = upcoming_appointments.count()
    except:
        upcoming_appointments = []
        upcoming_appointments_count = 0
    
    # Initialize context with safe values
    context = {
        'patient': patient,
        'upcoming_appointments': upcoming_appointments,
        'upcoming_appointments_count': upcoming_appointments_count,
        'recent_notifications': [],
        'unread_notifications_count': 0,
        'recent_prescriptions': [],
        'active_prescriptions_count': 0,
        'recent_medical_records': [],
        'total_pending_amount': 0
    }
    
    # Try to get notifications
    try:
        from notification.models import NotificationRecord
        recent_notifications = NotificationRecord.objects.filter(
            user=request.user
        ).order_by('-sent_at')[:5]
        unread_notifications_count = NotificationRecord.objects.filter(
            user=request.user, is_read=False
        ).count()
        
        context.update({
            'recent_notifications': recent_notifications,
            'unread_notifications_count': unread_notifications_count
        })
    except:
        # NotificationRecord model may not exist yet
        pass
    
    # Try to get prescription data
    try:
        from prescription.models import Prescription
        recent_prescriptions = Prescription.objects.filter(
            patient=patient,
            status__in=['pending', 'verified', 'dispensed']
        ).order_by('-date_prescribed')[:5]
        active_prescriptions_count = recent_prescriptions.count()
        
        context.update({
            'recent_prescriptions': recent_prescriptions,
            'active_prescriptions_count': active_prescriptions_count
        })
    except (ImportError, OperationalError):
        # Prescription model may not exist yet
        pass
    
    # Try to get medical records
    try:
        from medical_records.models import MedicalRecord
        recent_medical_records = MedicalRecord.objects.filter(
            patient=patient
        ).order_by('-record_date')[:5]
        
        context.update({
            'recent_medical_records': recent_medical_records
        })
    except (ImportError, OperationalError):
        # MedicalRecord model may not exist yet
        pass
    
    # Try to get billing data
    try:
        from billing.models import Bill
        # Fix the query to use fields that actually exist in the database
        unpaid_bills = Bill.objects.filter(
            patient=patient,
            status__in=['pending', 'overdue', 'partially_paid']
        )
        # Calculate total due manually from the bills
        total_pending_amount = sum(bill.total_amount - bill.amount_paid for bill in unpaid_bills)
        
        context.update({
            'total_pending_amount': total_pending_amount
        })
    except (ImportError, OperationalError):
        # Bill model may not exist yet
        pass
    
    # Get recent notifications using correct field names
    try:
        from notification.models import NotificationRecord
        
        # Get unread notifications count
        unread_notifications_count = NotificationRecord.objects.filter(
            user=request.user,
            read=False
        ).count()
        
        # Get recent notifications, prioritize unread ones
        user_notifications = NotificationRecord.objects.filter(
            user=request.user
        ).order_by('read', '-created_at')[:5]  # Unread first, then most recent
        
    except Exception as e:
        user_notifications = []
        unread_notifications_count = 0
        print(f"Error fetching notifications: {str(e)}")
    
    # Get recent vital signs from nurse module instead of medical
    try:
        from nurse.models import VitalSigns
        recent_vitals = VitalSigns.objects.filter(
            patient=patient
        ).order_by('-recorded_at')[:5]
    except Exception as e:
        recent_vitals = []
        print(f"Error fetching vital signs: {str(e)}")
    
    # Get recent lab test results from laboratory module instead of medical
    try:
        from laboratory.models import LabTest
        recent_lab_tests = LabTest.objects.filter(
            patient=patient
        ).order_by('-test_date')[:5]  # Using test_date instead of date_performed
    except Exception as e:
        recent_lab_tests = []
        print(f"Error fetching lab tests: {str(e)}")
    
    context.update({
        'user_notifications': user_notifications,
        'unread_notifications_count': unread_notifications_count,
        'recent_vitals': recent_vitals,
        'recent_lab_tests': recent_lab_tests,
    })
    
    return render(request, 'patient/dashboard.html', context)


@login_required
def chat_with_ai(request):
    """View for patient to chat with Healthcare AI Assistant using PyTorch model"""
    if not hasattr(request.user, 'is_patient') or not request.user.is_patient:
        messages.error(request, "Only patients can use the AI assistant.")
        return redirect('home')

    # Initialize HealthDiagnosisBot (ideally a singleton or initialized elsewhere)
    bot = HealthDiagnosisBot()

    # Check if we should reset the chat
    if 'reset' in request.GET:
        for key in ['chat_history', 'input_symptoms_simple', 'current_symptom_index', 'state', 'diagnosis_data']:
            if key in request.session:
                del request.session[key]
        logger.info("Chat history and symptom data reset")
        return redirect('patient:chat')

    # Get session data
    chat_history = request.session.get('chat_history', [])
    input_symptoms_simple = request.session.get('input_symptoms_simple', [])
    current_symptom_index = request.session.get('current_symptom_index', 0)
    state = request.session.get('state', 'initial')
    simple_symptoms = bot.simple_symptoms  # Get symptoms from bot

    # If this is a POST request, process the user's input
    if request.method == 'POST':
        user_message = request.POST.get('message', '').strip().lower().replace(" ", "")
        button_response = request.POST.get('button_response', '').lower()  # Handle button clicks
        logger.info(f"Received: message={user_message}, button_response={button_response}")

        # Prioritize button response if present, otherwise process text message
        if button_response in ['y', 'n']:
            user_input = button_response
            chat_history.append({"role": "user", "content": user_input.upper()})
        elif user_message:
            user_input = user_message
            chat_history.append({"role": "user", "content": user_input})
        else:
            user_input = None

        if user_input:
            try:
                if state == 'initial':
                    if user_message == 'chẩnđoán':
                        # Start diagnosis process
                        state = 'chat'
                        request.session['state'] = state
                        request.session['input_symptoms_simple'] = []
                        request.session['current_symptom_index'] = 0
                        ai_response = f"Bạn có triệu chứng {simple_symptoms[0]}? (Y/N):"
                        chat_history.append({"role": "assistant", "content": ai_response, "show_buttons": True})
                    else:
                        ai_response = (
                            "Chào bạn! Tôi là trợ lý sức khỏe AI.\n"
                            "Vui lòng trả lời các câu hỏi sau bằng Yes/No (Có/Không).\n"
                            "Hãy nói 'chẩnđoán' để bắt đầu!"
                        )
                        chat_history.append({"role": "assistant", "content": ai_response})

                elif state == 'chat':
                    # Process symptom input
                    if user_input in ['y', 'n']:
                        input_symptoms_simple.append(1 if user_input == 'y' else 0)
                        current_symptom_index += 1

                        if current_symptom_index < len(simple_symptoms):
                            # Ask next symptom
                            ai_response = f"Bạn có triệu chứng {simple_symptoms[current_symptom_index]}? (Y/N):"
                            chat_history.append({"role": "assistant", "content": ai_response, "show_buttons": True})
                        else:
                            # All symptoms collected, run diagnosis
                            input_symptoms_full = [0] * len(bot.symptom_names)
                            for i, simple_symptom in enumerate(simple_symptoms):
                                if simple_symptom in bot.symptom_names:
                                    idx = bot.symptom_names.index(simple_symptom)
                                    input_symptoms_full[idx] = input_symptoms_simple[i]

                            # Run diagnosis
                            mean_probs, std_probs, diagnosis, diagnosis_info = bot.diagnose(input_symptoms_full)

                            # Store diagnosis data in session for appointment booking
                            diagnosis_data = {
                                'diagnosis': diagnosis,
                                'severity': diagnosis_info['severity'],
                                'tests': diagnosis_info['tests'],
                                'treatments': diagnosis_info['treatments'],
                                'advice': diagnosis_info['advice']
                            }
                            request.session['diagnosis_data'] = diagnosis_data

                            # Format diagnosis results
                            ai_response = "\nKết quả chẩn đoán với xác suất và độ không chắc chắn:\n"
                            for i, dis in enumerate(bot.diseases):
                                ai_response += f"{dis}: Xác suất={mean_probs[0][i]:.3f}, Độ không chắc chắn={std_probs[0][i]:.3f}\n"
                            ai_response += (
                                f"\nChẩn đoán: {diagnosis} (±{std_probs[0][np.argmax(mean_probs)]:.3f})\n"
                                f"Mức độ nghiêm trọng: {diagnosis_info['severity']}\n"
                                f"Xét nghiệm đề xuất: {', '.join(diagnosis_info['tests'])}\n"
                                f"Thuốc/Điều trị đề xuất: {', '.join(diagnosis_info['treatments'])}\n"
                                f"Lời khuyên: {diagnosis_info['advice']}\n"
                                "\nLưu ý: Đây chỉ là chẩn đoán sơ bộ. Vui lòng tham khảo bác sĩ để được kiểm tra chính xác.\n\n"
                                "Bạn có muốn đặt lịch hẹn với bác sĩ để thảo luận về chẩn đoán này không? (Yes/No)"
                            )
                            chat_history.append({"role": "assistant", "content": ai_response, "show_buttons": True})

                            # Transition to booking decision state
                            state = 'booking_decision'
                            request.session['state'] = state

                        # Update session
                        request.session['input_symptoms_simple'] = input_symptoms_simple
                        request.session['current_symptom_index'] = current_symptom_index

                    else:
                        ai_response = "Vui lòng trả lời bằng 'Yes' hoặc 'No'."
                        chat_history.append({"role": "assistant", "content": ai_response, "show_buttons": True})

                elif state == 'booking_decision':
                    if user_input == 'y':
                        # Redirect to appointment booking page with diagnosis data
                        return redirect('patient:appointment_book')
                    elif user_input == 'n':
                        # Reset to initial state and show welcome message
                        state = 'initial'
                        input_symptoms_simple = []
                        current_symptom_index = 0
                        request.session['state'] = state
                        request.session['input_symptoms_simple'] = []
                        request.session['current_symptom_index'] = 0
                        if 'diagnosis_data' in request.session:
                            del request.session['diagnosis_data']
                        ai_response = (
                            "Chào bạn! Tôi là trợ lý sức khỏe AI.\n"
                            "Vui lòng trả lời các câu hỏi sau bằng Yes/No (Có/Không).\n"
                            "Hãy nói 'chẩnđoán' để bắt đầu!"
                        )
                        chat_history.append({"role": "assistant", "content": ai_response})
                    else:
                        ai_response = "Vui lòng trả lời bằng 'Yes' hoặc 'No'."
                        chat_history.append({"role": "assistant", "content": ai_response, "show_buttons": True})

                # Save chat history to session
                request.session['chat_history'] = chat_history
                request.session.modified = True

            except Exception as e:
                messages.error(request, "Có lỗi khi xử lý yêu cầu của bạn.")
                logger.error(f"AI Assistant error: {str(e)}", exc_info=True)
                ai_response = "Xin lỗi, tôi gặp sự cố. Bạn có muốn thử lại hoặc bắt đầu chẩn đoán không?"
                chat_history.append({"role": "assistant", "content": ai_response})
                request.session['chat_history'] = chat_history
                request.session.modified = True

    else:
        # GET request: Show initial message if state is initial
        if state == 'initial' and not chat_history:
            ai_response = (
                "Chào bạn! Tôi là trợ lý sức khỏe AI.\n"
                "Vui lòng trả lời các câu hỏi sau bằng Yes/No (Có/Không).\n"
                "Hãy nói 'chẩnđoán' để bắt đầu!"
            )
            chat_history.append({"role": "assistant", "content": ai_response})
            request.session['chat_history'] = chat_history
            request.session.modified = True

    return render(request, 'patient/chat.html', {
        'chat_history': chat_history,
    })
@login_required
def appointment_list(request):
    """View all appointments"""
    if not request.user.role == 'patient':
        return HttpResponseForbidden("You do not have access to this page.")
    
    try:
        patient = Patient.objects.get(user=request.user)
    except Patient.DoesNotExist:
        patient = Patient.objects.create(user=request.user)
    
    status = request.GET.get('status', '')
    date_from = request.GET.get('date_from', '')
    date_to = request.GET.get('date_to', '')
    
    appointments = Appointment.objects.filter(patient=patient).order_by('-appointment_date', '-appointment_time')
    
    if status:
        appointments = appointments.filter(status=status)
    
    if date_from:
        appointments = appointments.filter(appointment_date__gte=date_from)
    
    if date_to:
        appointments = appointments.filter(appointment_date__lte=date_to)
    
    context = {
        'appointments': appointments,
        'patient': patient,
    }
    return render(request, 'patient/appointments/list.html', context)

@login_required
def appointment_book(request):
    """Book a new appointment"""
    if not request.user.role == 'patient':
        return HttpResponseForbidden("You do not have access to this page.")
    
    try:
        patient = Patient.objects.get(user=request.user)
    except Patient.DoesNotExist:
        patient = Patient.objects.create(user=request.user)
    
    if request.method == 'POST':
        doctor_id = request.POST.get('doctor_id')
        appointment_date = request.POST.get('appointment_date')
        appointment_time = request.POST.get('appointment_time')
        reason = request.POST.get('reason')
        notes = request.POST.get('notes')
        
        # Validate required fields
        if not all([doctor_id, appointment_date, appointment_time, reason]):
            messages.error(request, "Please fill in all required fields")
            return redirect('patient:appointment_book')
            
        try:
            doctor = DoctorModel.objects.get(id=doctor_id)
            
            appointment = Appointment.objects.create(
                patient=patient,
                doctor=doctor,
                appointment_date=appointment_date,
                appointment_time=appointment_time,
                reason=reason,
                notes=notes,
                status='scheduled'
            )
            
            # Create notification record
            try:
                NotificationRecord.objects.create(
                    user=request.user,
                    subject='Appointment Scheduled',
                    message=f'Your appointment with Dr. {doctor.user.get_full_name()} on {appointment_date} at {appointment_time} has been scheduled.',
                    notification_type='appointment',
                    action_url=f'/patient/appointment/{appointment.id}/',
                    action_text='View Appointment',
                    created_at=timezone.now(),
                    read=False
                )
            except Exception as e:
                logger.warning(f"Failed to create notification: {str(e)}")
                
            # Clear diagnosis data from session after booking
            if 'diagnosis_data' in request.session:
                del request.session['diagnosis_data']
                
            messages.success(request, 'Appointment booked successfully!')
            return redirect('patient:appointment_detail', pk=appointment.id)
        except Exception as e:
            messages.error(request, f'Error booking appointment: {str(e)}')
    
    # Get all active doctors
    doctors = []
    setup_incomplete = True
    try:
        doctors = list(DoctorModel.objects.filter(user__is_active=True, accepting_patients=True))
        setup_incomplete = len(doctors) == 0
    except (ImportError, OperationalError) as e:
        messages.warning(request, "Doctor database is not yet set up. Please try again later or contact support.")
        logger.error(f"Error accessing doctors: {str(e)}")
    
    # Retrieve diagnosis data from session if available
    diagnosis_data = request.session.get('diagnosis_data', None)
    initial_reason = "Medical consultation"
    initial_notes = ""
    if diagnosis_data:
        initial_reason = f"Consultation for {diagnosis_data['diagnosis']}"
        initial_notes = (
            f"AI Diagnosis:\n"
            f"Diagnosis: {diagnosis_data['diagnosis']}\n"
            f"Severity: {diagnosis_data['severity']}\n"
            f"Recommended Tests: {', '.join(diagnosis_data['tests'])}\n"
            f"Recommended Treatments: {', '.join(diagnosis_data['treatments'])}\n"
            f"Advice: {diagnosis_data['advice']}"
        )

    # Set initial date to tomorrow
    initial_date = (timezone.now().date() + timedelta(days=1)).strftime('%Y-%m-%d')

    # Generate available dates (next 7 days)
    available_dates = [
        (timezone.now().date() + timedelta(days=i)).strftime('%Y-%m-%d')
        for i in range(1, 8)  # Start from tomorrow
    ]

    context = {
        'doctors': doctors,
        'min_date': timezone.now().date() + timedelta(days=1),
        'setup_incomplete': setup_incomplete,
        'initial_reason': initial_reason,
        'initial_notes': initial_notes,
        'initial_date': initial_date,
        'available_dates': available_dates
    }
    return render(request, 'patient/appointments/book.html', context)
@login_required
def appointment_detail(request, pk):
    """View appointment details"""
    if not request.user.role == 'patient':
        return HttpResponseForbidden("You do not have access to this page.")
    
    appointment = get_object_or_404(Appointment, pk=pk, patient__user=request.user)
    return render(request, 'patient/appointments/detail.html', {'appointment': appointment})

@login_required
def appointment_cancel(request, pk):
    """Cancel an appointment"""
    if not request.user.role == 'patient':
        return HttpResponseForbidden("You do not have access to this page.")
    
    appointment = get_object_or_404(Appointment, pk=pk, patient__user=request.user)
    
    if request.method == 'POST':
        # Only allow cancellation if appointment is scheduled or confirmed
        if appointment.status in ['scheduled', 'confirmed']:
            appointment.status = 'canceled'
            appointment.save()
            
            # Create notification
            try:
                NotificationRecord.objects.create(
                    user=request.user,
                    title='Appointment Canceled',
                    message=f'Your appointment with Dr. {appointment.doctor.user.get_full_name()} on {appointment.appointment_date} has been canceled.',
                    notification_type='appointment',
                    action_url=f'/patient/appointment/{appointment.id}/',
                    action_text='View Details'
                )
            except Exception:
                pass
                
            messages.success(request, 'Appointment canceled successfully.')
        else:
            messages.error(request, 'This appointment cannot be canceled.')
    
    return redirect('patient:appointment_detail', pk=appointment.id)

@login_required
def appointment_reschedule(request, pk):
    """Reschedule an appointment"""
    if not request.user.role == 'patient':
        return HttpResponseForbidden("You do not have access to this page.")
    
    appointment = get_object_or_404(Appointment, pk=pk, patient__user=request.user)
    
    if appointment.status not in ['scheduled', 'confirmed']:
        messages.error(request, 'This appointment cannot be rescheduled.')
        return redirect('patient:appointment_detail', pk=appointment.id)
    
    if request.method == 'POST':
        appointment_date = request.POST.get('appointment_date')
        
        appointment_time = request.POST.get('appointment_time')
        
        if not appointment_date or not appointment_time:
            messages.error(request, 'Please provide both date and time.')
        else:
            # Save the updated appointment details
            appointment.appointment_date = appointment_date
            appointment.appointment_time = appointment_time
            appointment.save()
            
            # Create notification
            try:
                NotificationRecord.objects.create(
                    user=request.user,
                    title='Appointment Rescheduled',
                    message=f'Your appointment with Dr. {appointment.doctor.user.get_full_name()} has been rescheduled to {appointment_date} at {appointment_time}.',
                    notification_type='appointment',
                    action_url=f'/patient/appointment/{appointment.id}/',
                    action_text='View Details'
                )
            except Exception:
                pass
                
            messages.success(request, 'Appointment rescheduled successfully.')
            return redirect('patient:appointment_detail', pk=appointment.id)
    
    context = {
        'appointment': appointment,
        'min_date': timezone.now().date(),
    }
    return render(request, 'patient/appointments/reschedule.html', context)

@login_required
def profile_view(request):
    """View patient profile"""
    if not request.user.role == 'patient':
        return HttpResponseForbidden("You do not have access to this page.")
    
    try:
        patient = Patient.objects.get(user=request.user)
    except Patient.DoesNotExist:
        patient = Patient.objects.create(user=request.user)
    
    context = {
        'user': request.user,
        'patient': patient
    }
    return render(request, 'patient/profile/view.html', context)

@login_required
def profile_edit(request):
    """Edit patient profile"""
    if not request.user.role == 'patient':
        return HttpResponseForbidden("You do not have access to this page.")
    
    try:
        patient = Patient.objects.get(user=request.user)
    except Patient.DoesNotExist:
        patient = Patient.objects.create(user=request.user)
    
    # Create the forms with initial data
    if request.method == 'POST':
        user_form = PatientUserForm(request.POST, instance=request.user)
        patient_form = PatientForm(request.POST, instance=patient)
        
        if user_form.is_valid() and patient_form.is_valid():
            user_form.save()
            patient_form.save()
            messages.success(request, "Profile updated successfully!")
            return redirect('patient:profile')
    else:
        user_form = PatientUserForm(instance=request.user)
        patient_form = PatientForm(instance=patient)
    
    context = {
        'user_form': user_form,
        'patient_form': patient_form,
    }
    return render(request, 'patient/profile/edit.html', context)

@login_required
def submit_insurance(request):
    """View to submit insurance information"""
    if not request.user.role == 'patient':
        return HttpResponseForbidden("You do not have access to this page.")
    
    try:
        patient = request.user.patient_profile
    except:
        messages.error(request, "Patient profile not found.")
        return redirect('patient:profile')
    
    # Check if there are any insurance providers in the system
    from insurance.models import InsuranceProvider
    providers_exist = InsuranceProvider.objects.exists()
    
    if not providers_exist:
        messages.warning(request, "No insurance providers are available in the system yet. Please contact administration.")
        return redirect('patient:profile')
    
    if request.method == 'POST':
        from insurance.forms import InsurancePolicyForm
        form = InsurancePolicyForm(request.POST, user=request.user)
        
        if form.is_valid():
            policy = form.save(commit=False)
            policy.patient = patient
            policy.save()
            
            # Update patient's primary insurance reference
            # Fix: Use provider name (string) instead of provider instance
            patient.insurance_provider = policy.provider.name
            patient.insurance_policy_number = policy.policy_number
            patient.save()
            
            messages.success(request, "Insurance information submitted successfully.")
            return redirect('patient:profile')
    else:
        from insurance.forms import InsurancePolicyForm
        # Initialize form with current user
        form = InsurancePolicyForm(user=request.user, initial={
            'patient': patient,
            'status': 'pending'  # Change from 'active' to 'pending' so the insurance provider can verify it
        })
    
    context = {
        'form': form,
        'patient': patient,
        'providers_exist': providers_exist
    }
    return render(request, 'patient/profile/insurance_form.html', context)

@login_required
def patient_policies(request):
    """View for patients to see their insurance policies"""
    if not request.user.is_patient:
        messages.error(request, "Only patients can view their policies.")
        return redirect('home')
    
    try:
        patient = request.user.patient_profile
        
        # Get all policies for this patient
        from insurance.models import InsurancePolicy
        policies = InsurancePolicy.objects.filter(patient=patient).order_by('-start_date')
        
        # Get active policy
        active_policy = policies.filter(status='active').first()
        
        # Get recent policy notifications
        from notification.models import NotificationRecord
        policy_notifications = NotificationRecord.objects.filter(
            user=request.user,
            notification_type='insurance_policy',
            read=False
        ).order_by('-created_at')[:3]
        
    except Exception as e:
        messages.error(request, f"Error loading your policies: {str(e)}")
        policies = []
        active_policy = None
        policy_notifications = []
    
    return render(request, 'patient/policies.html', {
        'policies': policies,
        'active_policy': active_policy,
        'policy_notifications': policy_notifications,
    })

@login_required
def medical_record_list(request):
    """View medical records including lab tests"""
    if not request.user.role == 'patient':
        return HttpResponseForbidden("You do not have access to this page.")
    
    try:
        patient = Patient.objects.get(user=request.user)
    except Patient.DoesNotExist:
        patient = Patient.objects.create(user=request.user)
    
    record_type = request.GET.get('record_type', '')
    date_from = request.GET.get('date_from', '')
    date_to = request.GET.get('date_to', '')
    search = request.GET.get('search', '')
    
    # Start with all records for this patient
    medical_records = MedicalRecord.objects.filter(patient=patient).order_by('-record_date')
    
    # Apply filters
    if record_type:
        medical_records = medical_records.filter(record_type=record_type)
    
    if date_from:
        # Update to use record_date instead of date
        medical_records = medical_records.filter(record_date__gte=date_from)
    
    if date_to:
        # Update to use record_date instead of date
        medical_records = medical_records.filter(record_date__lte=date_to)
    
    if search:
        medical_records = medical_records.filter(
            Q(description__icontains=search) |
            Q(doctor__user__first_name__icontains=search) |
            Q(doctor__user__last_name__icontains=search)
        )
    
    # Get lab tests
    try:
        from laboratory.models import LabTest
        lab_tests = LabTest.objects.filter(patient__user=request.user)
        
        # Apply date filter if provided
        date_filter = request.GET.get('date_filter', '')
        if date_filter:
            # Change requested_date to created_at
            lab_tests = lab_tests.filter(created_at__date=date_filter)
    except (ImportError, OperationalError):
        lab_tests = []
    
    context = {
        'medical_records': medical_records,
        'lab_tests': lab_tests,
        'patient': patient,
    }
    return render(request, 'patient/medical_records/list.html', context)

@login_required
def medical_record_detail(request, pk):
    """View medical record details"""
    if not request.user.role == 'patient':
        return HttpResponseForbidden("You do not have access to this page.")
    
    record = get_object_or_404(MedicalRecord, pk=pk, patient__user=request.user)
    return render(request, 'patient/medical_records/detail.html', {'record': record})

@login_required
def medical_record_pdf(request, pk):
    """Generate PDF for medical record"""
    if not request.user.role == 'patient':
        return HttpResponseForbidden("You do not have access to this page.")
    
    record = get_object_or_404(MedicalRecord, pk=pk, patient__user=request.user)
    
    # For now, just redirect back to the detail page with a message
    # In a real application, you would generate a PDF here
    messages.info(request, "PDF generation is not yet implemented. This feature will be available soon.")
    return redirect('patient:medical_record_detail', pk=pk)

@login_required
def prescription_list(request):
    """View prescriptions"""
    if not request.user.role == 'patient':
        return HttpResponseForbidden("You do not have access to this page.")
    
    try:
        patient = Patient.objects.get(user=request.user)
    except Patient.DoesNotExist:
        patient = Patient.objects.create(user=request.user)
    
    status = request.GET.get('status', '')
    date_from = request.GET.get('date_from', '')
    date_to = request.GET.get('date_to', '')
    
    # Initialize with empty list in case the table doesn't exist
    prescriptions = []
    
    # Try to get prescription data
    try:
        prescriptions = Prescription.objects.filter(patient=patient).order_by('-date_prescribed')
        
        if status:
            prescriptions = prescriptions.filter(status=status)
        
        if date_from:
            prescriptions = prescriptions.filter(date_prescribed__gte=date_from)
        
        if date_to:
            prescriptions = prescriptions.filter(date_prescribed__lte=date_to)
            
    except (OperationalError, Exception) as e:
        # Handle the case where the prescription table doesn't exist
        messages.warning(request, "Prescription functionality is not available at this time.")
    
    context = {
        'prescriptions': prescriptions,
        'patient': patient,
    }
    return render(request, 'patient/prescriptions/list.html', context)

@login_required
def prescription_detail(request, pk):
    """View prescription details"""
    if not request.user.role == 'patient':
        return HttpResponseForbidden("You do not have access to this page.")
    
    prescription = get_object_or_404(Prescription, pk=pk, patient__user=request.user)
    return render(request, 'patient/prescriptions/detail.html', {'prescription': prescription})

@login_required
def bill_list(request):
    """View bills and payments"""
    if not request.user.role == 'patient':
        return HttpResponseForbidden("You do not have access to this page.")
    
    try:
        patient = Patient.objects.get(user=request.user)
    except Patient.DoesNotExist:
        patient = Patient.objects.create(user=request.user)
    
    status = request.GET.get('status', '')
    bill_type = request.GET.get('bill_type', '')
    date_from = request.GET.get('date_from', '')
    date_to = request.GET.get('date_to', '')
    
    # Get unpaid bills
    unpaid_bills = Bill.objects.filter(
        patient=patient,
        status__in=['pending', 'overdue', 'partially_paid']
    ).order_by('due_date')
    
    # Calculate total due manually
    total_due = sum(bill.total_amount - bill.amount_paid for bill in unpaid_bills)
    
    # Get payments
    payments = Payment.objects.filter(patient=patient).order_by('-payment_date')
    
    if status:
        payments = payments.filter(status=status)
    
    if date_from:
        payments = payments.filter(payment_date__gte=date_from)
    
    if date_to:
        payments = payments.filter(payment_date__lte=date_to)
    
    # Get this month's payments
    this_month_start = timezone.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    paid_this_month = Payment.objects.filter(
        patient=patient,
        payment_date__gte=this_month_start,
        status='completed'
    ).aggregate(total=Sum('amount'))['total'] or 0
    
    # Get pending insurance amount
    pending_insurance = Bill.objects.filter(
        patient=patient,
        insurance_status='pending'
    ).aggregate(total=Sum('insurance_amount'))['total'] or 0
    
    payment_methods_count = PaymentMethod.objects.filter(user=request.user).count()
    
    context = {
        'unpaid_bills': unpaid_bills,
        'payments': payments,
        'total_due': total_due,
        'paid_this_month': paid_this_month,
        'pending_insurance': pending_insurance,
        'payment_methods_count': payment_methods_count,
        'patient': patient,
    }
    return render(request, 'patient/billing/list.html', context)

@login_required
def bill_detail(request, pk):
    """View bill details"""
    if not request.user.role == 'patient':
        return HttpResponseForbidden("You do not have access to this page.")
    
    bill = get_object_or_404(Bill, pk=pk, patient__user=request.user)
    return render(request, 'patient/billing/detail.html', {'bill': bill})

@login_required
def bill_pay(request, pk):
    """Pay a bill"""
    if not request.user.role == 'patient':
        return HttpResponseForbidden("You do not have access to this page.")
    
    bill = get_object_or_404(Bill, pk=pk, patient__user=request.user)
    return render(request, 'patient/billing/pay.html', {'bill': bill})

@login_required
def payment_methods(request):
    """View and manage payment methods"""
    if not request.user.role == 'patient':
        return HttpResponseForbidden("You do not have access to this page.")
    
    payment_methods = PaymentMethod.objects.filter(user=request.user)
    bank_accounts = payment_methods.filter(type='bank_account')
    card_methods = payment_methods.filter(type='credit_card')
    
    context = {
        'payment_methods': payment_methods,
        'bank_accounts': bank_accounts,
    }
    return render(request, 'patient/billing/payment_methods.html', context)

@login_required
def notifications(request):
    """View all notifications for the patient"""
    if not request.user.is_patient:
        messages.error(request, "Only patients can view notifications.")
        return redirect('home')
    
    try:
        from notification.models import NotificationRecord
        
        # Debug: Print all notifications in the system
        all_notifications = NotificationRecord.objects.all()
        print(f"Total notifications in system: {all_notifications.count()}")
        
        # Debug: Print notification types in system
        notification_types = NotificationRecord.objects.values_list('notification_type', flat=True).distinct()
        print(f"Notification types in system: {list(notification_types)}")
        
        # Debug: Print user info
        print(f"Current user: {request.user.username}, ID: {request.user.id}")
        
        # Get user's notifications - order by read status first, then by created_at
        notifications = NotificationRecord.objects.filter(user=request.user).order_by('read', '-created_at')
        print(f"User notifications: {notifications.count()}")
        
        # Print notifications for lab results specifically
        lab_notifications = notifications.filter(notification_type='lab_result')
        print(f"Lab result notifications: {lab_notifications.count()}")
        print(f"Unread lab notifications: {lab_notifications.filter(read=False).count()}")
        
        # List all notifications for this user
        for notification in notifications:
            print(f"ID: {notification.id}, Type: {notification.notification_type}, " 
                  f"Subject: {notification.subject}, Read: {notification.read}")
        
        # Mark notifications as read if requested
        if request.GET.get('mark_all_read'):
            notifications.update(read=True)
            messages.success(request, "All notifications marked as read.")
            return redirect('patient:notifications')
        
    except Exception as e:
        import traceback
        print(f"Error in notifications view: {str(e)}")
        print(traceback.format_exc())
        notifications = []
        messages.error(request, f"Could not load notifications: {str(e)}")
    
    return render(request, 'patient/notifications.html', {
        'notifications': notifications,
    })

@login_required
def notification_detail(request, pk):
    """View details of a specific notification"""
    if not request.user.is_patient:
        messages.error(request, "Only patients can view notifications.")
        return redirect('home')
    
    try:
        from notification.models import NotificationRecord
        notification = get_object_or_404(NotificationRecord, id=pk, user=request.user)
        
        # Mark as read if it's not already
        if not notification.read:
            notification.read = True
            notification.save()
        
    except Exception as e:
        messages.error(request, f"Could not load notification: {str(e)}")
        return redirect('patient:notifications')
    
    return render(request, 'patient/notification_detail.html', {
        'notification': notification,
    })

@login_required
def lab_test_detail(request, pk):
    """View lab test details as a patient"""
    if not request.user.role == 'patient':
        return HttpResponseForbidden("You do not have access to this page.")
    
    # Get test if it exists
    try:
        from laboratory.models import LabTest
        test = get_object_or_404(LabTest, pk=pk, patient__user=request.user)
    except (ImportError, OperationalError):
        messages.warning(request, "The laboratory system is still being set up.")
        return redirect('patient:dashboard')
    
    context = {
        'test': test
    }
    return render(request, 'patient/lab_tests/detail.html', context)

@login_required
def view_lab_result(request, lab_result_id):
    """View a specific lab test result"""
    if not request.user.role == 'patient':
        return HttpResponseForbidden("You do not have access to this page.")
    
    from laboratory.models import LabTest
    
    try:
        # Get the specific lab test for this patient
        lab_test = get_object_or_404(LabTest, id=lab_result_id, patient__user=request.user)
        
        # Get result items if available
        result_items = []
        if hasattr(lab_test, 'result_items'):
            result_items = lab_test.result_items.all()
        
        context = {
            'lab_test': lab_test,
            'result_items': result_items
        }
        return render(request, 'patient/lab_results/detail.html', context)
        
    except Exception as e:
        messages.error(request, f"Error retrieving lab test: {str(e)}")
        return redirect('patient:dashboard')

@login_required
def invoice_detail(request, pk):
    """View invoice details"""
    if not request.user.role == 'patient':
        return HttpResponseForbidden("You do not have access to this page.")
    
    try:
        from pharmacy.models import PrescriptionInvoice
        invoice = get_object_or_404(PrescriptionInvoice, id=pk, patient__user=request.user)
        
        # Calculate medication costs for display
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
            pass
        
    except ImportError:
        messages.error(request, "The invoice system is not available.")
        return redirect('patient:dashboard')
        
    context = {
        'invoice': invoice,
        'medication_costs': medication_costs,
    }
    return render(request, 'patient/invoices/detail.html', context)

@login_required
def bills(request):
    """View and pay bills"""
    if not request.user.role == 'patient':
        return HttpResponseForbidden("You don't have access to this page.")
    
    try:
        patient = Patient.objects.get(user=request.user)
    except Patient.DoesNotExist:
        patient = Patient.objects.create(user=request.user)
    
    # Process payment if form submitted
    if request.method == 'POST':
        # Check for regular bill payment
        bill_id = request.POST.get('bill_id')
        if bill_id:
            try:
                from billing.models import Bill
                bill = get_object_or_404(Bill, id=bill_id, patient=patient)
                
                # Process payment (in a real app, this would integrate with a payment gateway)
                bill.mark_as_paid()
                
                messages.success(request, f"Payment of ${bill.amount} successfully processed.")
                
                # Create a payment notification
                NotificationRecord.objects.create(
                    user=request.user,
                    subject="Payment Confirmation",
                    message=f"Your payment of ${bill.amount} for invoice #{bill.invoice_number} was successfully processed.",
                    notification_type="payment",
                    action_url="/patient/bills/",  # Use action_url instead of target_url
                    action_text="View Bills"
                )
                
                return redirect('patient:bills')
            except Exception as e:
                messages.error(request, f"Error processing bill payment: {str(e)}")
        
        # Check for pharmacy invoice payment
        invoice_id = request.POST.get('invoice_id')
        if invoice_id:
            try:
                # Import models inside the try block to handle module unavailability
                from pharmacy.models import PrescriptionInvoice
                
                # Process the payment
                invoice = get_object_or_404(PrescriptionInvoice, id=invoice_id, patient=patient)
                
                # Make sure the invoice isn't already paid
                if invoice.status == 'paid':
                    messages.info(request, f"Invoice #{invoice.invoice_number} has already been paid.")
                    return redirect('patient:bills')
                
                # Mark as paid
                invoice.mark_as_paid()
                
                messages.success(request, f"Payment of ${invoice.total_amount} successfully processed.")
                
                # Create a payment notification
                NotificationRecord.objects.create(
                    user=request.user,
                    subject="Prescription Payment Confirmation",
                    message=f"Your payment of ${invoice.total_amount} for prescription invoice #{invoice.invoice_number} was successfully processed.",
                    notification_type="payment",
                    action_url="/patient/bills/",
                    action_text="View Bills"
                )
                
                return redirect('patient:bills')
            except ImportError:
                messages.error(request, "Pharmacy module is not available.")
            except Exception as e:
                messages.error(request, f"Error processing prescription payment: {str(e)}")
    
    # Initialize lists and amounts
    pending_bills = []
    paid_bills = []
    total_pending_amount = 0
    total_paid_amount = 0
    
    # Get regular bills
    try:
        from billing.models import Bill
        
        # Get all bills for this patient, ordered by due date
        regular_bills = Bill.objects.filter(patient=patient).order_by('due_date')
        
        # Mark any overdue bills
        for bill in regular_bills.filter(status='pending', due_date__lt=timezone.now().date()):
            bill.status = 'overdue'
            bill.save(update_fields=['status'])
        
        # Get counts and totals after potential status updates
        pending_bills = list(regular_bills.filter(status__in=['pending', 'overdue']))
        paid_bills = list(regular_bills.filter(status='paid'))
        
        # Calculate totals for regular bills
        total_pending_amount += sum(bill.amount for bill in pending_bills)
        total_paid_amount += sum(bill.paid_amount for bill in paid_bills)
        
    except (ImportError, OperationalError):
        # Handle case where billing module doesn't exist
        pass
    
    # Get pharmacy invoices
    try:
        from pharmacy.models import PrescriptionInvoice
        from payment.models import Payment
        
        # Get prescription invoices for this patient
        rx_invoices = PrescriptionInvoice.objects.filter(patient=patient).order_by('due_date')
        
        # Add pending pharmacy invoices to pending bills
        pending_rx = list(rx_invoices.filter(status='pending'))
        pending_bills.extend(pending_rx)
        total_pending_amount += sum(invoice.total_amount for invoice in pending_rx)
        
        # Add paid pharmacy invoices to paid bills
        paid_rx = list(rx_invoices.filter(status='paid'))
        paid_bills.extend(paid_rx)
        total_paid_amount += sum(invoice.total_amount for invoice in paid_rx)
        
        # Get payment history
        payments = Payment.objects.filter(patient=patient).order_by('-payment_date')
        
    except (ImportError, OperationalError):
        # Handle case where pharmacy module doesn't exist
        payments = []
    
    context = {
        'pending_bills': pending_bills,
        'paid_bills': paid_bills,
        'payments': payments if 'payments' in locals() else [],
        'total_pending_amount': total_pending_amount,
        'total_paid_amount': total_paid_amount,
        'patient': patient,
    }
    return render(request, 'patient/bills/list.html', context)

@login_required
def view_vitals(request, vitals_id):
    """View vital signs and care notes"""
    if not request.user.role == 'patient':
        return HttpResponseForbidden("You do not have access to this page.")
    
    try:
        from nurse.models import VitalSigns
        vitals = get_object_or_404(VitalSigns, id=vitals_id, patient__user=request.user)
    except ImportError:
        messages.error(request, "The vitals module is not available.")
        return redirect('patient:dashboard')
        
    context = {
        'vitals': vitals
    }
    return render(request, 'patient/vitals/detail.html', context)

@login_required
def health_records(request):
    """View for patient to see their health records, including vitals and lab tests"""
    if not request.user.is_patient:
        messages.error(request, "Only patients can view their health records.")
        return redirect('home')
    
    try:
        patient = request.user.patient_profile
        
        # Get vital signs from nurse module
        from nurse.models import VitalSigns
        vitals = VitalSigns.objects.filter(
            patient=patient
        ).order_by('-recorded_at')
        
        # Get lab test results from laboratory module
        from laboratory.models import LabTest
        lab_tests = LabTest.objects.filter(
            patient=patient
        ).order_by('-requested_date')  # Changed to order by requested_date (newest first)
        
    except ImportError as e:
        messages.error(request, f"Error loading your health records: {str(e)}")
        vitals = []
        lab_tests = []
    except Exception as e:
        messages.error(request, f"Error loading your health records: {str(e)}")
        vitals = []
        lab_tests = []
    
    return render(request, 'patient/health_records.html', {
        'vitals': vitals,
        'lab_tests': lab_tests,
    })

@login_required
def check_notifications(request):
    """Ajax endpoint to check for new notifications"""
    if not request.user.is_patient:
        return JsonResponse({"error": "Not authorized"}, status=403)
    
    try:
        from notification.models import NotificationRecord
        unread_count = NotificationRecord.objects.filter(
            user=request.user,
            read=False
        ).count()
        
        # Get the most recent notifications
        recent = NotificationRecord.objects.filter(
            user=request.user
        ).order_by('-created_at')[:3]
        
        recent_data = [{
            'id': n.id,
            'subject': n.subject,
            'message': n.message[:50] + '...' if len(n.message) > 50 else n.message,
            'time': n.created_at.strftime('%Y-%m-%d %H:%M'),
            'action_url': n.action_url,  # Use action_url instead of target_url
            'action_text': n.action_text
        } for n in recent]
        
        return JsonResponse({
            "unread_count": unread_count,
            "recent": recent_data,
            "success": True
        })
    except Exception as e:
        import traceback
        print(f"Notification check error: {str(e)}")
        print(traceback.format_exc())
        return JsonResponse({
            "error": str(e),
            "success": False
        })
