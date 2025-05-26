from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.utils import timezone
from django.db.models import Q, Sum
from django.views.generic import ListView
from .models import InsuranceProvider, InsurancePolicy, InsuranceClaim
from .forms import InsurancePolicyForm, InsuranceClaimForm
from patient.models import Patient


@login_required
def policy_list(request):
    """View for listing insurance policies"""
    user = request.user
    
    # Filter policies based on user role
    if user.is_patient:
        try:
            patient = user.patient_profile
            policies = InsurancePolicy.objects.filter(patient=patient)
        except:
            policies = InsurancePolicy.objects.none()
            messages.error(request, "Patient profile not found.")
    elif user.is_insurance or user.is_admin or user.is_superuser:
        policies = InsurancePolicy.objects.all()
    else:
        policies = InsurancePolicy.objects.none()
        messages.error(request, "You don't have permission to view insurance policies.")
    
    # Apply filters if provided
    status = request.GET.get('status')
    if status:
        policies = policies.filter(status=status)
    
    search = request.GET.get('search')
    if search:
        policies = policies.filter(
            Q(policy_number__icontains=search) |
            Q(patient__user__first_name__icontains=search) |
            Q(patient__user__last_name__icontains=search) |
            Q(provider__name__icontains=search)
        )
    
    context = {
        'policies': policies,
        'status_filter': status,
        'search_term': search,
    }
    
    return render(request, 'insurance/policy_list.html', context)


@login_required
def policy_detail(request, pk):
    """View for insurance policy details"""
    policy = get_object_or_404(InsurancePolicy, pk=pk)
    
    # Check permissions
    user = request.user
    if user.is_patient and policy.patient.user != user:
        messages.error(request, "You don't have permission to view this policy.")
        return redirect('insurance:policy_list')
    
    # Process actions (approve/reject) for pending policies
    if request.method == 'POST' and (user.role == 'insurance' or user.is_admin or user.is_superuser):
        action = request.POST.get('action')
        notes = request.POST.get('notes', '')
        
        # Debug message to verify POST is being received
        messages.info(request, f"Processing {action} action for policy {policy.policy_number}")
        
        if action == 'approve' and policy.status == 'pending':
            # Update policy status
            InsurancePolicy.objects.filter(pk=policy.id).update(status='active')
            
            # Verify the change happened
            policy.refresh_from_db()
            messages.success(request, f"Policy {policy.policy_number} has been approved successfully. New status: {policy.status}")
            
            # Send notification to patient
            try:
                from notification.models import NotificationRecord
                
                NotificationRecord.objects.create(
                    user=policy.patient.user,
                    subject="Insurance Policy Approved",
                    message=f"Your insurance policy #{policy.policy_number} has been approved and is now active. " +
                            f"Provider: {policy.provider.name}. " +
                            (f"Notes: {notes}" if notes else ""),
                    notification_type="insurance_policy",
                    action_url="/patient/policies/",
                    action_text="View Policies",
                    read=False
                )
                
                messages.info(request, "Notification sent to patient.")
                
            except Exception as e:
                messages.warning(request, f"Could not send notification: {str(e)}")
            
            # Important: Redirect to force a fresh page load
            return redirect('insurance:policy_detail', pk=policy.id)
            
        elif action == 'reject' and policy.status == 'pending':
            # Update policy status
            InsurancePolicy.objects.filter(pk=policy.id).update(status='canceled')
            
            # Verify the change happened
            policy.refresh_from_db()
            messages.warning(request, f"Policy {policy.policy_number} has been rejected. New status: {policy.status}")
            
            # Send notification to patient
            try:
                from notification.models import NotificationRecord
                
                NotificationRecord.objects.create(
                    user=policy.patient.user,
                    subject="Insurance Policy Rejected",
                    message=f"Your insurance policy #{policy.policy_number} has been rejected. " +
                            f"Please contact your insurance provider ({policy.provider.name}) for more information. " +
                            (f"Reason: {notes}" if notes else ""),
                    notification_type="insurance_policy",
                    action_url="/patient/help/",
                    action_text="Contact Support",
                    read=False
                )
                
                messages.info(request, "Notification sent to patient.")
                
            except Exception as e:
                messages.warning(request, f"Could not send notification: {str(e)}")
            
            # Important: Redirect to force a fresh page load
            return redirect('insurance:policy_detail', pk=policy.id)
    
    # Get related claims
    claims = InsuranceClaim.objects.filter(insurance_policy=policy).order_by('-claim_date')
    
    return render(request, 'insurance/policy_detail.html', {
        'policy': policy,
        'claims': claims,
    })


@login_required
def create_claim(request, policy_id=None):
    """View for creating a new insurance claim"""
    policy = None
    if policy_id:
        policy = get_object_or_404(InsurancePolicy, pk=policy_id)
        
        # Check permissions
        if request.user.is_patient and policy.patient.user != request.user:
            messages.error(request, "You don't have permission to create claims for this policy.")
            return redirect('policy_list')
    
    if request.method == 'POST':
        form = InsuranceClaimForm(request.POST)
        
        if form.is_valid():
            claim = form.save()
            messages.success(request, f"Claim {claim.claim_number} created successfully")
            return redirect('claim_detail', pk=claim.id)
    else:
        initial = {}
        if policy:
            initial['insurance_policy'] = policy
            initial['patient'] = policy.patient
            
        form = InsuranceClaimForm(initial=initial)
        
        # Customize form based on user role
        if request.user.is_patient:
            try:
                patient = request.user.patient_profile
                form.fields['patient'].initial = patient
                form.fields['patient'].widget.attrs['disabled'] = True
                form.fields['insurance_policy'].queryset = InsurancePolicy.objects.filter(patient=patient)
            except:
                pass
    
    return render(request, 'insurance/create_claim.html', {
        'form': form,
        'policy': policy,
    })


@login_required
def process_claim(request, pk):
    """View for processing an insurance claim"""
    claim = get_object_or_404(InsuranceClaim, pk=pk)
    
    # Check permissions
    if not (request.user.is_insurance or request.user.is_admin):
        messages.error(request, "You don't have permission to process insurance claims.")
        return redirect('claim_list')
    
    if request.method == 'POST':
        approval_status = request.POST.get('approval_status')
        if approval_status in ['approved', 'rejected', 'partial']:
            claim.approval_status = approval_status
            
            if approval_status in ['approved', 'partial']:
                try:
                    approved_amount_str = request.POST.get('approved_amount', '0')
                    approved_amount = float(approved_amount_str)
                    claim.approved_amount = approved_amount
                    
                    # Apply discount to the patient's bill
                    try:
                        from billing.models import Bill
                        bills = Bill.objects.filter(
                            patient=claim.patient,
                            status__in=['pending', 'overdue', 'partially_paid']
                        ).order_by('-date_created')
                        
                        if bills.exists():
                            bill = bills.first()
                            bill.insurance_amount = approved_amount
                            bill.amount_due = max(0, bill.total_amount - bill.insurance_amount)
                            bill.insurance_status = 'approved'
                            bill.save()
                    except (ImportError, Exception) as e:
                        messages.warning(request, f"Could not apply insurance discount to bill: {str(e)}")
                        
                except ValueError as e:
                    messages.error(request, f"Invalid approved amount: {str(e)}")
                    return redirect('insurance:process_claim', pk=claim.id)
            else:
                try:
                    from billing.models import Bill
                    bills = Bill.objects.filter(
                        patient=claim.patient,
                        status__in=['pending', 'overdue', 'partially_paid']
                    ).order_by('-date_created')
                    
                    if bills.exists():
                        bill = bills.first()
                        bill.insurance_status = 'rejected'
                        bill.save()
                except Exception as e:
                    messages.warning(request, f"Could not update bill: {str(e)}")
            
            claim.notes = request.POST.get('notes', '')
            claim.processed_date = timezone.now()
            claim.processed_by = request.user
            claim.save()
            
            # Enhanced notification for the patient
            try:
                from notification.models import NotificationRecord
                
                if approval_status == 'approved':
                    title = "Insurance Claim Approved"
                    message = f"Your insurance claim #{claim.claim_number} has been approved. The approved amount of ${claim.approved_amount} will be applied to your bill."
                elif approval_status == 'partial':
                    title = "Insurance Claim Partially Approved"
                    message = f"Your insurance claim #{claim.claim_number} has been partially approved. The approved amount of ${claim.approved_amount} will be applied to your bill."
                else:
                    title = "Insurance Claim Rejected"
                    message = f"Your insurance claim #{claim.claim_number} has been rejected. Please contact our billing department for more information."
                
                if claim.notes:
                    message += f"\n\nAdditional notes: {claim.notes}"
                
                notification = NotificationRecord.objects.create(
                    user=claim.patient.user,
                    subject=title,  # Use subject instead of title
                    message=message,
                    notification_type="insurance",
                    action_url="/patient/bills/",
                    action_text="View Bills",
                    read=False  # Use read instead of is_read
                )
                
                try:
                    from django.core.mail import send_mail
                    from django.conf import settings
                    
                    if claim.patient.user.email:
                        send_mail(
                            subject=title,
                            message=message,
                            from_email=settings.DEFAULT_FROM_EMAIL,
                            recipient_list=[claim.patient.user.email],
                            fail_silently=True
                        )
                except Exception as email_error:
                    print(f"Failed to send email notification: {str(email_error)}")
                
                print(f"Notification created successfully: ID={notification.id}")
                messages.success(request, f"Claim processed and notification sent to patient.")
                
            except Exception as e:
                messages.warning(request, f"Could not create notification: {str(e)}")
                
            messages.success(request, f"Claim {claim.claim_number} processed successfully")
            
            if request.POST.get('from_dashboard'):
                return redirect('insurance:dashboard')
            return redirect('insurance:claim_detail', pk=claim.id)
        else:
            messages.error(request, f"Invalid approval status: {approval_status}")
    
    return render(request, 'insurance/process_claim.html', {
        'claim': claim,
    })


class InsuranceClaimListView(LoginRequiredMixin, ListView):
    """View for listing insurance claims"""
    model = InsuranceClaim
    template_name = 'insurance/claim_list.html'
    context_object_name = 'claims'
    paginate_by = 10
    
    def get_queryset(self):
        queryset = InsuranceClaim.objects.all().order_by('-claim_date')
        user = self.request.user
        
        # Filter by user role
        if user.is_patient:
            try:
                queryset = queryset.filter(patient__user=user)
            except:
                queryset = InsuranceClaim.objects.none()
        
        # Filter by status if provided
        status = self.request.GET.get('status')
        if status:
            queryset = queryset.filter(approval_status=status)
            
        # Filter by search term
        search = self.request.GET.get('search')
        if search:
            queryset = queryset.filter(
                Q(claim_number__icontains=search) |
                Q(patient__user__first_name__icontains=search) |
                Q(patient__user__last_name__icontains=search)
            )
            
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['status_filter'] = self.request.GET.get('status', '')
        context['search_term'] = self.request.GET.get('search', '')
        context['status_choices'] = InsuranceClaim.APPROVAL_STATUS_CHOICES
        return context


@login_required
def claim_detail(request, pk):
    """View for insurance claim details"""
    claim = get_object_or_404(InsuranceClaim, pk=pk)
    
    # Check permissions
    user = request.user
    if user.is_patient and claim.patient.user != user:
        messages.error(request, "You don't have permission to view this claim.")
        return redirect('claim_list')
    
    return render(request, 'insurance/claim_detail.html', {
        'claim': claim,
    })


@login_required
def insurance_dashboard(request):
    """Display the insurance dashboard with pending claims"""
    # Check if user has insurance privileges
    if not (request.user.is_insurance or request.user.is_admin):
        messages.error(request, "You don't have permission to access the insurance dashboard.")
        return redirect('home')
    
    # Get all claims first to diagnose the issue
    all_claims = InsuranceClaim.objects.all()
    
    # Get a list of all existing policies
    all_policies = InsurancePolicy.objects.all()
    pending_policies = all_policies.filter(status='pending')
    active_policies = all_policies.filter(status='active')
    
    # Activate an existing policy if requested
    if request.GET.get('activate_policy'):
        policy_id = request.GET.get('activate_policy')
        try:
            policy = InsurancePolicy.objects.get(id=policy_id)
            
            # Fix: Don't rely on patient.insurance_policy attribute
            # Simply update the policy status directly
            policy.status = 'active'
            
            # Use custom save to avoid any model-level issues
            InsurancePolicy.objects.filter(id=policy_id).update(status='active')
            
            messages.success(request, f"Policy {policy.policy_number} has been activated successfully")
            
            # Refresh from database to get updated values
            policy.refresh_from_db()
            
            # Send notification to patient
            try:
                from notification.models import NotificationRecord
                
                NotificationRecord.objects.create(
                    user=policy.patient.user,
                    subject="Insurance Policy Approved",
                    message=f"Your insurance policy #{policy.policy_number} has been approved and is now active. " +
                            f"Provider: {policy.provider.name}.",
                    notification_type="insurance_policy",
                    action_url="/patient/policies/",
                    action_text="View Policies",
                    read=False
                )
                
                messages.info(request, "Notification sent to patient.")
                
            except Exception as e:
                messages.warning(request, f"Could not send notification: {str(e)}")
                
        except InsurancePolicy.DoesNotExist:
            messages.error(request, f"Policy with ID {policy_id} not found")
        except Exception as e:
            messages.error(request, f"Error activating policy: {str(e)}")
    
    # Create test policy if requested
    if request.GET.get('create_test_policy'):
        try:
            # Find a patient to associate with policy
            from patient.models import Patient
            patients = Patient.objects.all()
            if patients.exists():
                patient = patients.first()
                
                # Find or create an insurance provider
                providers = InsuranceProvider.objects.all()
                if providers.exists():
                    provider = providers.first()
                else:
                    # Create a basic provider with only required fields
                    provider = InsuranceProvider.objects.create(
                        name="Test Insurance Co."
                    )
                
                # Create a test policy
                import random
                import string
                from django.utils import timezone
                
                today = timezone.now().date()
                policy_number = f"POL-{''.join(random.choices(string.ascii_uppercase + string.digits, k=8))}"
                
                policy = InsurancePolicy.objects.create(
                    patient=patient,
                    provider=provider,
                    policy_number=policy_number,
                    member_id=f"MEM-{random.randint(10000, 99999)}",
                    group_number=f"GRP-{random.randint(100, 999)}",
                    start_date=today,
                    end_date=today.replace(year=today.year + 1),
                    status='active',
                    coverage_percentage=80,
                    coverage_details="Standard health coverage",
                    deductible=1000.00,
                    co_pay=20.00,
                    out_of_pocket_max=5000.00
                )
                
                messages.success(request, f"Created test policy #{policy.policy_number} for {patient.user.get_full_name()}")
            else:
                messages.error(request, "No patients found in the system to create policy.")
        except Exception as e:
            messages.error(request, f"Could not create test policy: {str(e)}")
    
    # Create test claim if requested
    if request.GET.get('create_test_claim'):
        try:
            # Get active policies
            policies = InsurancePolicy.objects.filter(status='active')
            
            if policies.exists():
                policy = policies.first()
                patient = policy.patient
                
                # Create a test claim
                test_claim = InsuranceClaim.objects.create(
                    patient=patient,
                    insurance_policy=policy,
                    service_date=timezone.now().date(),
                    claim_amount=500.00,
                    approval_status='pending',
                    diagnosis_codes="R10.9",
                    service_codes="99213",
                    notes="Test claim for demonstration"
                )
                messages.success(request, f"Created test claim #{test_claim.claim_number} for {patient.user.get_full_name()}")
            else:
                messages.error(request, "No active policies found. Create a test policy first.")
        except Exception as e:
            messages.error(request, f"Could not create test claim: {str(e)}")
    
    # Re-query pending claims after potential test data creation
    pending_claims = InsuranceClaim.objects.filter(approval_status='pending').order_by('-claim_date')
    
    # Re-query policies after changes
    all_policies = InsurancePolicy.objects.all()
    pending_policies = all_policies.filter(status='pending')
    active_policies = all_policies.filter(status='active')
    
    context = {
        'pending_claims': pending_claims,
        'pending_count': pending_claims.count(),
        'all_claims_count': all_claims.count(),
        'active_policies': active_policies,
        'pending_policies': pending_policies,
        'all_policies_count': all_policies.count(),
        'has_active_policies': active_policies.exists(),
    }
    
    return render(request, 'insurance/dashboard.html', context)


@login_required
def verification_request_list(request):
    """View for listing insurance verification requests"""
    # Update the permission check to recognize both 'insurance' and 'insurer' roles
    if not (request.user.role == 'insurance' or request.user.role == 'insurer' or 
            getattr(request.user, 'is_insurance', False)):
        messages.error(request, "You don't have permission to view verification requests.")
        return redirect('home')
    
    # Get verification requests (policies with pending status)
    verification_requests = InsurancePolicy.objects.filter(status='pending').order_by('-created_at')
    
    # Apply filters if provided
    search = request.GET.get('search')
    if search:
        verification_requests = verification_requests.filter(
            Q(policy_number__icontains=search) |
            Q(patient__user__first_name__icontains=search) |
            Q(patient__user__last_name__icontains=search) |
            Q(provider__name__icontains=search)
        )
    
    context = {
        'verification_requests': verification_requests,
        'search_term': search,
    }
    
    return render(request, 'insurance/verification_requests.html', context)


class InsurancePolicyListView(LoginRequiredMixin, ListView):
    """View for listing insurance policies"""
    model = InsurancePolicy
    template_name = 'insurance/policy_list.html'
    context_object_name = 'policies'
    paginate_by = 10
    
    def get_queryset(self):
        queryset = InsurancePolicy.objects.all().order_by('-start_date')
        user = self.request.user
        
        # Filter by user role
        if user.is_patient:
            try:
                queryset = queryset.filter(patient__user=user)
            except:
                queryset = InsurancePolicy.objects.none()
        elif user.is_insurance:
            # Insurance providers can see all policies
            pass
        elif not (user.is_admin or user.is_superuser):
            # Other users cannot see policies unless they're admins
            queryset = InsurancePolicy.objects.none()
            
        # Filter by status if provided
        status = self.request.GET.get('status')
        if status:
            queryset = queryset.filter(status=status)
            
        # Filter by search term
        search = self.request.GET.get('search')
        if search:
            queryset = queryset.filter(
                Q(policy_number__icontains=search) |
                Q(patient__user__first_name__icontains=search) |
                Q(patient__user__last_name__icontains=search) |
                Q(provider__name__icontains=search)
            )
            
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['status_filter'] = self.request.GET.get('status', '')
        context['search_term'] = self.request.GET.get('search', '')
        context['status_choices'] = InsurancePolicy.STATUS_CHOICES
        return context