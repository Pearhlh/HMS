from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .services import send_email_notification, send_sms_notification

@login_required
def notification_settings(request):
    """View for user's notification preferences"""
    if request.method == 'POST':
        # Update notification settings
        user = request.user
        user.email_notifications = request.POST.get('email_notifications') == 'on'
        user.sms_notifications = request.POST.get('sms_notifications') == 'on'
        user.save()
        
        messages.success(request, "Notification settings updated successfully")
        return redirect('notification_settings')
    
    return render(request, 'notification/settings.html')

@login_required
def notification_history(request):
    """View for user's notification history"""
    # In a real system, we would track notifications sent to users
    # Here we'll just return a template with empty data
    return render(request, 'notification/history.html', {'notifications': []})

@login_required
def test_notification(request):
    """View for testing notification delivery"""
    if request.method == 'POST':
        notification_type = request.POST.get('type', 'email')
        
        if notification_type == 'email':
            result = send_email_notification(
                request.user.email,
                "Test Notification",
                'notifications/test_notification.html',
                {'name': request.user.get_full_name() or request.user.email}
            )
            
            if result:
                messages.success(request, f"Test email sent to {request.user.email}")
            else:
                messages.error(request, "Failed to send test email")
                
        elif notification_type == 'sms' and request.user.phone_number:
            result = send_sms_notification(
                request.user.phone_number,
                f"This is a test message from Healthcare System for {request.user.get_full_name() or request.user.email}"
            )
            
            if result:
                messages.success(request, f"Test SMS sent to {request.user.phone_number}")
            else:
                messages.error(request, "Failed to send test SMS")
        else:
            messages.error(request, "Phone number not available for SMS testing")
    
    return render(request, 'notification/test_notification.html')
