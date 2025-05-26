from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth import get_user_model
from django.views.generic import UpdateView, DetailView
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.urls import reverse_lazy
from django.contrib import messages
from django.db.models import Q
from .forms import UserProfileForm, RoleSpecificProfileForm, AdminUserCreationForm, AdminUserUpdateForm, UserFilterForm, SetPasswordForm
from .models import User


def is_admin_or_superuser(user):
    """Check if user is admin or superuser"""
    return user.is_admin or user.is_superuser


def login_success(request):
    """
    Redirects users based on their role after login
    """
    if request.user.is_authenticated:
        if request.user.role == 'patient':
            return redirect('patient:dashboard')
        elif request.user.role == 'doctor':
            return redirect('doctor:dashboard')
        elif request.user.role == 'nurse':
            return redirect('nurse:dashboard')
        elif request.user.role == 'admin':
            return redirect('admin:index')
        elif request.user.role in ['insurance', 'insurer']:  # Handle both role names
            return redirect('insurance:dashboard')
        elif request.user.role == 'pharmacist':
            return redirect('pharmacy:dashboard')
        elif request.user.role == 'laboratory':
            return redirect('laboratory:dashboard')
    
    # Default fallback
    return redirect('home')


@login_required
def profile_view(request):
    """View for user profile"""
    user = request.user
    
    # Get role-specific profile if it exists
    role_profile = None
    role_form = None
    
    if user.is_patient:
        from patient.models import Patient
        role_profile = hasattr(user, 'patient_profile') and user.patient_profile
        from patient.forms import PatientProfileForm
        role_form_class = PatientProfileForm
    elif user.is_doctor:
        from doctor.models import Doctor
        role_profile = hasattr(user, 'doctor_profile') and user.doctor_profile
        from doctor.forms import DoctorProfileForm
        role_form_class = DoctorProfileForm
    else:
        role_form_class = None
    
    if request.method == 'POST':
        user_form = UserProfileForm(request.POST, request.FILES, instance=user)
        if role_profile and role_form_class:
            role_form = role_form_class(request.POST, instance=role_profile)
            if user_form.is_valid() and role_form.is_valid():
                user_form.save()
                role_form.save()
                messages.success(request, 'Your profile was successfully updated!')
                return redirect('profile')
        else:
            if user_form.is_valid():
                user_form.save()
                messages.success(request, 'Your profile was successfully updated!')
                return redirect('profile')
    else:
        user_form = UserProfileForm(instance=user)
        if role_profile and role_form_class:
            role_form = role_form_class(instance=role_profile)
    
    context = {
        'user_form': user_form,
        'role_form': role_form,
    }
    return render(request, 'users/profile.html', context)


@login_required
@user_passes_test(is_admin_or_superuser)
def user_management(request):
    """View for user management dashboard"""
    # Count users by role
    user_counts = {
        'total': User.objects.count(),
        'active': User.objects.filter(is_active=True).count(),
        'inactive': User.objects.filter(is_active=False).count(),
        'patients': User.objects.filter(role='patient').count(),
        'doctors': User.objects.filter(role='doctor').count(),
        'nurses': User.objects.filter(role='nurse').count(),
        'admins': User.objects.filter(role='admin').count(),
        'pharmacists': User.objects.filter(role='pharmacist').count(),
        'insurances': User.objects.filter(role='insurance').count(),
        'lab_techs': User.objects.filter(role='lab_technician').count(),
    }
    
    # Get recent users
    recent_users = User.objects.all().order_by('-date_joined')[:10]
    
    context = {
        'user_counts': user_counts,
        'recent_users': recent_users,
    }
    return render(request, 'users/user_management.html', context)


@login_required
@user_passes_test(is_admin_or_superuser)
def user_list(request):
    """View for listing users with filtering"""
    filter_form = UserFilterForm(request.GET)
    queryset = User.objects.all().order_by('-date_joined')
    
    if filter_form.is_valid():
        role = filter_form.cleaned_data.get('role')
        status = filter_form.cleaned_data.get('status')
        search = filter_form.cleaned_data.get('search')
        
        if role:
            queryset = queryset.filter(role=role)
        
        if status:
            is_active = (status == 'active')
            queryset = queryset.filter(is_active=is_active)
            
        if search:
            queryset = queryset.filter(
                Q(first_name__icontains=search) |
                Q(last_name__icontains=search) |
                Q(email__icontains=search)
            )
    
    context = {
        'users': queryset,
        'filter_form': filter_form
    }
    return render(request, 'users/user_list.html', context)


@login_required
@user_passes_test(is_admin_or_superuser)
def create_user(request):
    """View for creating a new user"""
    if request.method == 'POST':
        form = AdminUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(request, f"User {user.email} created successfully.")
            return redirect('user_detail', pk=user.id)
    else:
        form = AdminUserCreationForm()
    
    return render(request, 'users/create_user.html', {
        'form': form,
    })


@login_required
@user_passes_test(is_admin_or_superuser)
def update_user(request, pk):
    """View for updating an existing user"""
    user = get_object_or_404(User, pk=pk)
    
    if request.method == 'POST':
        form = AdminUserUpdateForm(request.POST, request.FILES, instance=user)
        if form.is_valid():
            user = form.save()
            messages.success(request, f"User {user.email} updated successfully.")
            return redirect('user_detail', pk=user.id)
    else:
        form = AdminUserUpdateForm(instance=user)
    
    return render(request, 'users/update_user.html', {
        'form': form,
        'user': user,
    })


@login_required
@user_passes_test(is_admin_or_superuser)
def user_detail(request, pk):
    """View for user details"""
    user = get_object_or_404(User, pk=pk)
    
    # Get role-specific profile information
    role_profile = None
    if user.role == 'patient':
        if hasattr(user, 'patient_profile'):
            role_profile = user.patient_profile
    elif user.role == 'doctor':
        if hasattr(user, 'doctor_profile'):
            role_profile = user.doctor_profile
    
    # Get recent activity
    # This would come from an activity tracking system
    # For now we'll just send empty data
    recent_activity = []
    
    context = {
        'user_obj': user,  # Using user_obj to avoid conflict with request.user
        'role_profile': role_profile,
        'recent_activity': recent_activity,
    }
    return render(request, 'users/user_detail.html', context)


@login_required
@user_passes_test(is_admin_or_superuser)
def set_user_password(request, pk):
    """View for setting a new password for a user"""
    user = get_object_or_404(User, pk=pk)
    
    if request.method == 'POST':
        form = SetPasswordForm(request.POST)
        if form.is_valid():
            password = form.cleaned_data['password1']
            user.set_password(password)
            user.save()
            messages.success(request, f"Password updated for {user.email}.")
            return redirect('user_detail', pk=user.id)
    else:
        form = SetPasswordForm()
    
    context = {
        'form': form,
        'user_obj': user,
    }
    return render(request, 'users/set_password.html', context)


@login_required
@user_passes_test(is_admin_or_superuser)
def toggle_user_status(request, pk):
    """View for activating/deactivating a user"""
    user = get_object_or_404(User, pk=pk)
    
    if request.method == 'POST':
        user.is_active = not user.is_active
        user.save()
        
        status = "activated" if user.is_active else "deactivated"
        messages.success(request, f"User {user.email} has been {status}.")
        return redirect('user_detail', pk=user.id)
    
    context = {
        'user_obj': user,
    }
    return render(request, 'users/confirm_status_change.html', context)


class AdminDashboardView(LoginRequiredMixin, UserPassesTestMixin):
    """Admin dashboard view for system management"""
    template_name = 'users/admin_dashboard.html'
    
    def test_func(self):
        return self.request.user.is_admin or self.request.user.is_superuser
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Count statistics for the dashboard
        from patient.models import Patient
        from doctor.models import Doctor
        from appointment.models import Appointment
        from prescription.models import Prescription
        
        context['total_patients'] = Patient.objects.count()
        context['total_doctors'] = Doctor.objects.count()
        context['total_appointments'] = Appointment.objects.count()
        context['recent_appointments'] = Appointment.objects.order_by('-appointment_date')[:5]
        context['total_prescriptions'] = Prescription.objects.count()
        
        return context
