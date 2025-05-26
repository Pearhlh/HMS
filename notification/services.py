from datetime import timezone
from django.conf import settings
from django.core.mail import send_mail
from django.template.loader import render_to_string
import logging

logger = logging.getLogger(__name__)

# Handle the missing twilio dependency
TWILIO_AVAILABLE = False
try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    # Twilio not installed, SMS functionality will be limited
    pass

def send_email_notification(recipient, subject, template, context):
    """Send email notification using Django's email system"""
    try:
        html_message = render_to_string(template, context)
        plain_message = render_to_string(template.replace('.html', '.txt'), context)
        
        send_mail(
            subject=subject,
            message=plain_message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[recipient],
            html_message=html_message,
            fail_silently=False
        )
        logger.info(f"Email notification sent to {recipient}")
        return True
    except Exception as e:
        logger.error(f"Email sending failed: {e}")
        return False

def send_sms_notification(phone_number, message):
    """Send SMS notification using Twilio"""
    if not TWILIO_AVAILABLE:
        logger.warning(f"SMS to {phone_number} not sent (Twilio not installed): {message}")
        return False
    
    try:
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=message,
            from_=settings.TWILIO_PHONE_NUMBER,
            to=phone_number
        )
        logger.info(f"SMS notification sent to {phone_number}")
        return True
    except Exception as e:
        logger.error(f"SMS sending failed: {e}")
        return False

def send_appointment_notification(appointment):
    """Send notification about appointment status changes"""
    patient = appointment.patient
    doctor = appointment.doctor
    
    # Different messages based on status
    if appointment.status == 'scheduled':
        subject = 'Appointment Scheduled'
        message = f"Your appointment with Dr. {doctor.user.get_full_name()} has been scheduled for {appointment.appointment_date} at {appointment.appointment_time}."
    elif appointment.status == 'confirmed':
        subject = 'Appointment Confirmed'
        message = f"Your appointment with Dr. {doctor.user.get_full_name()} on {appointment.appointment_date} at {appointment.appointment_time} has been confirmed."
    elif appointment.status == 'completed':
        subject = 'Appointment Completed'
        message = f"Thank you for visiting Dr. {doctor.user.get_full_name()}. We hope your visit was satisfactory."
    elif appointment.status == 'canceled':
        subject = 'Appointment Canceled'
        message = f"Your appointment with Dr. {doctor.user.get_full_name()} on {appointment.appointment_date} has been canceled."
    else:
        subject = 'Appointment Update'
        message = f"Your appointment status has been updated to {appointment.get_status_display()}."
    
    # Send email if patient has email
    if patient.user.email:
        send_email_notification(
            patient.user.email,
            subject,
            'notifications/appointment_update.html',
            {'appointment': appointment, 'message': message}
        )
    
    # Send SMS if patient has phone
    if hasattr(patient.user, 'phone_number') and patient.user.phone_number:
        send_sms_notification(patient.user.phone_number, message)
    
    return True

def send_prescription_notification(prescription):
    """Send notification about prescription status changes"""
    # Implement notification logic based on prescription status
    patient = prescription.patient
    
    if prescription.status == 'verified':
        subject = 'Prescription Verified'
        message = f"Your prescription #{prescription.id} has been verified and is ready to be dispensed."
    elif prescription.status == 'dispensed':
        subject = 'Prescription Dispensed'
        message = f"Your prescription #{prescription.id} has been dispensed. Please collect your medication."
    else:
        subject = 'Prescription Update'
        message = f"Your prescription #{prescription.id} status is now {prescription.get_status_display()}."
    
    # Send email if patient has email
    if patient.user.email:
        send_email_notification(
            patient.user.email,
            subject,
            'notifications/prescription_update.html',
            {'prescription': prescription, 'message': message}
        )
    
    # Send SMS if patient has phone
    if hasattr(patient.user, 'phone_number') and patient.user.phone_number:
        send_sms_notification(patient.user.phone_number, message)
    
    return True

def send_lab_test_notification(lab_test):
    """Send notification about lab test status changes"""
    patient = lab_test.patient
    
    if lab_test.status == 'completed':
        subject = 'Lab Test Results Ready'
        message = f"Your {lab_test.get_test_type_display()} test results are now available. Please consult with your doctor."
    else:
        subject = 'Lab Test Update'
        message = f"Your {lab_test.get_test_type_display()} test status is now {lab_test.get_status_display()}."
    
    # Log the attempt to create notification
    print(f"Attempting to create notification for lab test {lab_test.id} for patient {patient.user.username if patient and patient.user else 'unknown'}")
    
    # Create notification record in database
    try:
        from notification.models import NotificationRecord
        from django.utils import timezone
        
        # Make sure patient and user exist
        if not patient or not patient.user:
            print("❌ Patient or patient user is missing")
            return False
        
        # Create notification for patient using correct field names
        notification = NotificationRecord.objects.create(
            user=patient.user,
            subject=subject,
            message=message,
            notification_type='lab_result',
            created_at=timezone.now(),
            read=False,
            action_url=f"/patient/lab-results/{lab_test.id}/", # Direct link to results
            action_text="View Results"
        )
        
        print(f"✅ Created lab test notification for patient {patient.user.username}, record ID: {notification.id}")
        
        # Also notify doctor if there is one
        if hasattr(lab_test, 'requested_by') and lab_test.requested_by and lab_test.requested_by.user:
            doctor_notification = NotificationRecord.objects.create(
                user=lab_test.requested_by.user,
                subject=f"Lab Results for {patient.user.get_full_name()}",
                message=f"Lab test results for {patient.user.get_full_name()} are now available.",
                notification_type='lab_result',
                created_at=timezone.now(),
                read=False,
                action_url=f"/doctor/patient/{patient.id}/lab-test/{lab_test.id}/",
                action_text="View Results"
            )
            print(f"✅ Created lab test notification for doctor {lab_test.requested_by.user.username}, record ID: {doctor_notification.id}")
    except Exception as e:
        import traceback
        print(f"❌ Failed to create notification record: {str(e)}")
        print(traceback.format_exc())
        # Re-raise so calling code can handle the error
        raise
    
    # Send email if patient has email
    if patient.user.email:
        try:
            send_email_notification(
                patient.user.email,
                subject,
                'notifications/lab_test_update.html',
                {'lab_test': lab_test, 'message': message}
            )
            print(f"✅ Sent email notification to {patient.user.email}")
        except Exception as e:
            print(f"❌ Failed to send email: {str(e)}")
    else:
        print("ℹ️ Patient has no email address, skipping email notification")
    
    return True

def send_insurance_claim_notification(claim):
    """Send notification about insurance claim status changes"""
    patient = claim.patient
    
    if claim.approval_status == 'approved':
        subject = 'Insurance Claim Approved'
        message = f"Your insurance claim #{claim.claim_number} has been approved for ${claim.approved_amount}."
    elif claim.approval_status == 'partial':
        subject = 'Insurance Claim Partially Approved'
        message = f"Your insurance claim #{claim.claim_number} has been partially approved for ${claim.approved_amount}."
    elif claim.approval_status == 'rejected':
        subject = 'Insurance Claim Rejected'
        message = f"Your insurance claim #{claim.claim_number} has been rejected. Please contact your insurance provider for details."
    else:
        subject = 'Insurance Claim Update'
        message = f"Your insurance claim #{claim.claim_number} status is now {claim.get_approval_status_display()}."
    
    # Send email if patient has email
    if patient.user.email:
        send_email_notification(
            patient.user.email,
            subject,
            'notifications/insurance_claim_update.html',
            {'claim': claim, 'message': message}
        )
    
    # Send SMS if patient has phone
    if hasattr(patient.user, 'phone_number') and patient.user.phone_number:
        send_sms_notification(patient.user.phone_number, message)
    
    return True

def send_payment_confirmation(payment):
    """Send payment confirmation notification"""
    invoice = payment.invoice
    patient = invoice.patient
    
    subject = 'Payment Confirmation'
    message = f"We've received your payment of ${payment.amount} for Invoice #{invoice.invoice_number}. Thank you!"
    
    # Additional info if invoice is fully paid
    if invoice.is_paid:
        message += " Your invoice has been fully paid."
    else:
        message += f" Your remaining balance is ${invoice.amount_due}."
    
    # Send email if patient has email
    if patient.user.email:
        send_email_notification(
            patient.user.email,
            subject,
            'notifications/payment_confirmation.html',
            {'payment': payment, 'invoice': invoice, 'message': message}
        )
    
    # Send SMS if patient has phone
    if hasattr(patient.user, 'phone_number') and patient.user.phone_number:
        send_sms_notification(patient.user.phone_number, message)
    
    return True
