from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import HttpResponseForbidden
from django.contrib import messages
from django.utils import timezone
from django.db.utils import OperationalError
from django.urls import reverse
from django.db.models import Q

from .models import LabTest, TestResult, LabReport, LabResultItem
from .forms import LabTestForm, TestResultForm, LabResultItemForm, LabResultItemFormSet
from patient.models import Patient
from doctor.models import Doctor
from notification.models import NotificationRecord

def is_lab_staff(user):
    """Check if user is lab staff or admin"""
    return user.is_lab_technician or user.is_admin or user.is_superuser or user.role == 'lab_tech'

@login_required
@user_passes_test(is_lab_staff)
def lab_dashboard(request):
    """Laboratory dashboard view showing test statistics"""
    today = timezone.now().date()
    pending_tests = LabTest.objects.filter(status__in=['ordered', 'collected'])
    samples_to_collect = LabTest.objects.filter(status='ordered').count()
    in_process_tests = LabTest.objects.filter(status='in_process').count()
    completed_today = LabTest.objects.filter(
        status='completed',
        updated_at__date=today
    ).count()
    completed_tests = LabTest.objects.filter(status='completed').order_by('-updated_at')[:10]
    
    context = {
        'pending_tests': pending_tests,
        'samples_to_collect': samples_to_collect,
        'in_process_tests': in_process_tests,
        'completed_today': completed_today,
        'completed_tests': completed_tests,
    }
    
    return render(request, 'laboratory/dashboard.html', context)

@login_required
@user_passes_test(is_lab_staff)
def lab_test_list(request):
    """List of lab tests with filtering"""
    # Implementation goes here
    return render(request, 'laboratory/test_list.html', {'tests': []})

@login_required
@user_passes_test(is_lab_staff)
def process_lab_test(request, pk):
    """Process a laboratory test and add results"""
    # Implementation goes here
    return render(request, 'laboratory/process_test.html', {'lab_test': None})

@login_required
def view_lab_test(request, pk):
    return render(request, 'laboratory/view_test.html', {})

@login_required
@user_passes_test(is_lab_staff)
def register_sample(request):
    return render(request, 'laboratory/register_sample.html', {})

@login_required
@user_passes_test(is_lab_staff)
def export_test_results(request):
    return render(request, 'laboratory/export.html', {})

@login_required
@user_passes_test(is_lab_staff)
def test_equipment(request):
    return render(request, 'laboratory/equipment.html', {})

@login_required
@user_passes_test(is_lab_staff)
def lab_analytics(request):
    return render(request, 'laboratory/analytics.html', {})

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
        
        # Create the lab test request with explicit request date
        lab_test = LabTest.objects.create(
            patient=patient,
            requested_by=doctor,
            test_type=test_type,
            description=description,
            instructions=instructions,
            status='requested',
            requested_date=timezone.now()  # Explicitly set the requested date
        )
        
        messages.success(request, f"Test request for {lab_test.get_test_type_display()} has been submitted to the laboratory.")
        return redirect('doctor:patient_detail', pk=patient_id)
    
    context = {
        'patient': patient,
    }
    return render(request, 'laboratory/request_test.html', context)

@login_required
def dashboard(request):
    """Laboratory technician dashboard"""
    # Use a more user-friendly access check
    if not hasattr(request.user, 'role') or request.user.role != 'lab_tech':
        return render(request, 'access_denied.html', {
            'required_role': 'Lab Technician (lab_tech)'
        })
    
    # Get actual test requests from the database
    try:
        # Add debug info
        from .models import LabTest
        all_tests = LabTest.objects.all()
        print(f"Total lab tests in database: {all_tests.count()}")
        for test in all_tests:
            print(f"Test ID: {test.id}, Type: {test.test_type}, Patient: {test.patient}, Status: {test.status}")
        
        pending_tests = LabTest.objects.filter(status__in=['requested', 'scheduled']).order_by('-requested_date')[:5]
        print(f"Pending tests: {pending_tests.count()}")
        
        in_progress_tests = LabTest.objects.filter(status='in_progress').order_by('-requested_date')[:5]
        completed_tests = LabTest.objects.filter(status='completed').order_by('-completed_date')[:5]
        
        pending_count = LabTest.objects.filter(status='requested').count()
        in_progress_count = LabTest.objects.filter(status='in_progress').count()
        completed_today_count = LabTest.objects.filter(status='completed', 
                                                    completed_date__date=timezone.now().date()).count()
    except Exception as e:
        # Print detailed error for debugging
        import traceback
        print(f"Error querying lab tests: {str(e)}")
        print(traceback.format_exc())
        
        # Set default values
        pending_tests = []
        in_progress_tests = []
        completed_tests = []
        pending_count = 0
        in_progress_count = 0
        completed_today_count = 0
    
    context = {
        'user': request.user,
        'pending_tests': pending_tests,
        'in_progress_tests': in_progress_tests,
        'completed_tests': completed_tests,
        'pending_count': pending_count,
        'in_progress_count': in_progress_count,
        'completed_today_count': completed_today_count,
    }
    return render(request, 'laboratory/dashboard.html', context)

@login_required
def test_list(request):
    """View all lab test requests"""
    # Use a more user-friendly access check
    if not hasattr(request.user, 'role') or request.user.role != 'lab_tech':
        return render(request, 'access_denied.html', {
            'required_role': 'Lab Technician (lab_tech)'
        })
    
    # Get test status filter
    status = request.GET.get('status', '')
    
    # Debug: Print the status parameter
    print(f"Status filter from request: '{status}'")
    
    try:
        # Try to use ORM first for reliability
        from .models import LabTest
        
        if status:
            query = LabTest.objects.filter(status=status).order_by('-requested_date')
            print(f"ORM query found {query.count()} tests with status '{status}'")
        else:
            query = LabTest.objects.all().order_by('-requested_date')
            print(f"ORM query found {query.count()} total tests")
        
        tests = []
        for test in query:
            tests.append({
                'id': test.id,
                'patient': test.patient,
                'test_type': test.test_type,
                'status': test.status,
                'requested_date': test.requested_date,
                'description': test.description,
                'priority': getattr(test, 'priority', 'normal'),
                'get_status_display': test.get_status_display,
                'get_test_type_display': test.get_test_type_display,
            })
            
        # If ORM is empty but we should have data, fall back to raw SQL
        if not tests:
            # Use our safe utility function instead of ORM
            from .utils import get_tests_safely
            tests = get_tests_safely(status if status else None)
            
    except Exception as e:
        import traceback
        print(f"Error retrieving tests: {str(e)}")
        print(traceback.format_exc())
        tests = []
        messages.warning(request, "There was an issue retrieving test data.")
    
    # Debug: Print how many tests we're sending to the template
    print(f"Sending {len(tests)} tests to template")
    
    context = {
        'tests': tests,
        'current_status': status,
    }
    return render(request, 'laboratory/test_list.html', context)

@login_required
def test_detail(request, test_id):
    """View test details and add results"""
    # Use a more user-friendly access check
    if not hasattr(request.user, 'role') or request.user.role != 'lab_tech':
        return render(request, 'access_denied.html', {
            'required_role': 'Lab Technician (lab_tech)'
        })
    
    try:
        test = get_object_or_404(LabTest, pk=test_id)
        
        if request.method == 'POST':
            form = TestResultForm(request.POST, instance=test)
            if form.is_valid():
                previous_status = test.status
                test = form.save(commit=False)
                if test.status == 'completed' and not test.completed_date:
                    test.completed_date = timezone.now()
                    test.assigned_to = request.user
                test.save()
                
                # Check if status changed to completed
                if test.status == 'completed' and previous_status != 'completed':
                    # Import the service to send notifications
                    try:
                        # Use direct notification creation instead of the service for now
                        from notification.models import NotificationRecord
                        
                        # Create notification for patient
                        if test.patient and test.patient.user:
                            NotificationRecord.objects.create(
                                user=test.patient.user,
                                subject='Lab Test Results Available',
                                message=f'Your {test.get_test_type_display()} test results are now available.',
                                notification_type='lab_result',
                                created_at=timezone.now(),
                                read=False,
                                action_url=f"/patient/lab-results/{test.id}/",
                                action_text="View Results"
                            )
                            print(f"✅ Created notification for patient {test.patient.user.username} for test {test.id}")
                            
                            # Notify the doctor - using hardcoded URLs to avoid reverse errors
                            if test.requested_by and test.requested_by.user:
                                NotificationRecord.objects.create(
                                    user=test.requested_by.user,
                                    subject='Lab Test Results Available',
                                    message=f'Results for {test.get_test_type_display()} for patient {test.patient.user.get_full_name()} are now available.',
                                    notification_type='lab_result',
                                    created_at=timezone.now(),
                                    read=False,
                                    action_url=f'/doctor/patient/{test.patient.id}/lab-test/{test.id}/',
                                    action_text='View Results'
                                )
                                print(f"✅ Created notification for doctor {test.requested_by.user.username}")
                            
                            messages.success(request, "Test results saved and notifications sent successfully.")
                        else:
                            messages.warning(request, "Results saved but patient user not found for notification.")
                    except Exception as e:
                        import traceback
                        print(f"❌ Error creating notifications: {str(e)}")
                        print(traceback.format_exc())
                        messages.warning(request, "Results saved but notification failed. Check logs.")
                else:
                    messages.success(request, "Test results saved successfully.")
                    
                return redirect('laboratory:test_detail', test_id=test.id)
        else:
            # Make sure we're creating a form even for GET requests
            form = TestResultForm(instance=test)
            
    except Exception as e:
        # Handle all exceptions more gracefully
        print(f"Error retrieving lab test: {str(e)}")
        messages.error(request, f"An error occurred: {str(e)}")
        return redirect('laboratory:dashboard')
    
    context = {
        'test': test,
        'form': form
    }
    return render(request, 'laboratory/test_detail.html', context)

@login_required
def update_test_status(request, test_id):
    """Update the status of a test"""
    # Use a more user-friendly access check
    if not hasattr(request.user, 'role') or request.user.role != 'lab_tech':
        return render(request, 'access_denied.html', {
            'required_role': 'Lab Technician (lab_tech)'
        })
    
    try:
        test = get_object_or_404(LabTest, pk=test_id)
        
        if request.method == 'POST':
            new_status = request.POST.get('status')
            # Keep track of old status to detect changes
            old_status = test.status
            
            if new_status in dict(LabTest.TEST_STATUS_CHOICES):
                test.status = new_status
                if new_status == 'in_progress' and not test.assigned_to:
                    test.assigned_to = request.user
                elif new_status == 'completed' and not test.completed_date:
                    test.completed_date = timezone.now()
                
                # Save test results if provided
                results = request.POST.get('results', '')
                if results:
                    test.results = results
                    print(f"✅ Saving test results: {results[:50]}...")
                
                test.save()
                
                # Check if the status has changed to completed
                if new_status == 'completed' and old_status != 'completed':
                    # Send notification to patient
                    try:
                        # Import here to avoid circular imports
                        from notification.models import NotificationRecord
                        
                        # Create a notification record directly for the patient
                        if test.patient and test.patient.user:
                            NotificationRecord.objects.create(
                                user=test.patient.user,
                                subject='Lab Test Results Available',
                                message=f'Your {test.get_test_type_display()} results are now available for review.',
                                notification_type='lab_result',
                                created_at=timezone.now(),
                                read=False,
                                action_url=f"/patient/lab-results/{test.id}/",
                                action_text="View Results"
                            )
                            print(f"✅ Created notification for patient {test.patient.user.username} for test {test.id}")
                            
                            # Also create notification for the doctor if available
                            if test.requested_by and test.requested_by.user:
                                NotificationRecord.objects.create(
                                    user=test.requested_by.user,
                                    subject=f'Lab Test Results for {test.patient.user.get_full_name()}',
                                    message=f'Results for {test.get_test_type_display()} are now available.',
                                    notification_type='lab_result',
                                    created_at=timezone.now(),
                                    read=False,
                                    action_url=f"/doctor/patient/{test.patient.id}/lab-test/{test.id}/",
                                    action_text="View Results"
                                )
                                print(f"✅ Created notification for doctor {test.requested_by.user.username}")
                            
                            messages.success(request, f"Test status updated to {test.get_status_display()} and notifications sent.")
                        else:
                            messages.warning(request, "Test status updated but patient user not found.")
                    except Exception as e:
                        import traceback
                        print(f"❌ Error creating notification: {str(e)}")
                        print(traceback.format_exc())
                        messages.warning(request, f"Test status updated but notification failed: {str(e)}")
                else:
                    messages.success(request, f"Test status updated to {test.get_status_display()}")
            else:
                messages.error(request, "Invalid status")
                
        return redirect('laboratory:test_detail', test_id=test.id)
    except Exception as e:
        import traceback
        print(f"Error updating test status: {str(e)}")
        print(traceback.format_exc())
        messages.error(request, f"An error occurred: {str(e)}")
        return redirect('laboratory:dashboard')

@login_required
def record_results(request, test_id):
    """Record lab test results"""
    # Get the lab test
    lab_test = get_object_or_404(LabTest, id=test_id)
    
    if request.method == 'POST':
        # Process form submission
        form = LabTestForm(request.POST, instance=lab_test)
        formset = LabResultItemFormSet(request.POST, instance=lab_test)
        
        if form.is_valid() and formset.is_valid():
            # Update lab test status
            lab_test = form.save(commit=False)
            lab_test.status = 'completed'
            lab_test.results_date = timezone.now()
            lab_test.save()
            
            # Save test result items
            formset.save()
            
            # Create explicit debug message to verify this code is executing
            messages.info(request, "About to create notification for patient")
            
            # Notify patient with proper database field values
            try:
                from notification.models import NotificationRecord
                from django.utils import timezone
                from notification.services import send_lab_test_notification
                
                # Use the service function instead of direct creation
                send_lab_test_notification(lab_test)
                
                messages.success(request, f"Lab results recorded and notification sent to {lab_test.patient.user.username}")
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"NOTIFICATION ERROR: {str(e)}\n{error_details}")
                messages.error(request, f"Could not send notification: {str(e)}")
            
            return redirect('laboratory:test_detail', test_id=lab_test.id)
    else:
        # Display form
        form = LabTestForm(instance=lab_test)
        formset = LabResultItemFormSet(instance=lab_test)
    
    context = {
        'lab_test': lab_test,
        'form': form,
        'formset': formset
    }
    return render(request, 'laboratory/record_results.html', context)

@login_required
def finalize_lab_result(request, test_id):
    """Finalize lab test results and notify the patient"""
    # Use a more user-friendly access check
    if not hasattr(request.user, 'role') or request.user.role != 'lab_tech':
        return render(request, 'access_denied.html', {
            'required_role': 'Lab Technician (lab_tech)'
        })
    
    lab_test = get_object_or_404(LabTest, id=test_id)
    
    # Make sure the test is actually completed
    if lab_test.status != 'completed':
        messages.warning(request, "Test must be marked as completed before sending notifications.")
        return redirect('laboratory:test_detail', test_id=lab_test.id)
    
    # Update the notification fields to match the correct schema
    from django.utils import timezone
    
    try:
        # Create notification directly for more control
        from notification.models import NotificationRecord
        
        # Create patient notification
        if lab_test.patient and lab_test.patient.user:
            patient_notification = NotificationRecord.objects.create(
                user=lab_test.patient.user,
                subject='Lab Test Results Available',
                message=f'Your {lab_test.get_test_type_display()} test results are now available.',
                notification_type='lab_result',
                created_at=timezone.now(),
                read=False,
                action_url=f"/patient/lab-results/{lab_test.id}/",
                action_text="View Results"
            )
            print(f"✅ Created notification for patient {lab_test.patient.user.username}, ID: {patient_notification.id}")
            
            # Create doctor notification if applicable
            if lab_test.requested_by and lab_test.requested_by.user:
                doctor_notification = NotificationRecord.objects.create(
                    user=lab_test.requested_by.user,
                    subject=f'Lab Results for {lab_test.patient.user.get_full_name()}',
                    message=f'Results for {lab_test.get_test_type_display()} test for patient {lab_test.patient.user.get_full_name()} are now available.',
                    notification_type='lab_result',
                    created_at=timezone.now(),
                    read=False,
                    action_url=f"/doctor/patient/{lab_test.patient.id}/lab-test/{lab_test.id}/",
                    action_text="View Results"
                )
                print(f"✅ Created notification for doctor {lab_test.requested_by.user.username}, ID: {doctor_notification.id}")
            
            messages.success(request, f"Notification sent to {lab_test.patient.user.get_full_name()}")
        else:
            messages.error(request, "Patient or patient user not found. Notification not sent.")
    except Exception as e:
        import traceback
        print(f"❌ Error sending notification: {str(e)}")
        print(traceback.format_exc())
        messages.error(request, f"Failed to send notification: {str(e)}")
    
    return redirect('laboratory:test_detail', test_id=lab_test.id)
