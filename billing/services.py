from django.utils import timezone
from datetime import timedelta
from decimal import Decimal
from .models import Invoice
from insurance.models import InsurancePolicy, InsuranceClaim

def generate_prescription_invoice(dispensation):
    """Generate an invoice for dispensed medications"""
    # Calculate medication costs
    total_cost = Decimal('0.00')
    for item in dispensation.items.all():
        if item.inventory:
            item_cost = item.inventory.unit_price * item.quantity_dispensed
            total_cost += item_cost
    
    # Apply tax (could be configurable)
    tax_rate = Decimal('0.08')  # 8% tax
    tax_amount = total_cost * tax_rate
    
    # Check for insurance coverage
    insurance_coverage = Decimal('0.00')
    patient = dispensation.patient
    
    try:
        # Find active insurance policy
        today = timezone.now().date()
        policy = InsurancePolicy.objects.filter(
            patient=patient,
            status='active',
            start_date__lte=today,
            end_date__gte=today
        ).first()
        
        if policy:
            # Calculate coverage based on policy percentage
            coverage_percentage = Decimal(policy.coverage_percentage) / 100
            insurance_coverage = total_cost * coverage_percentage
            
            # Create insurance claim record
            InsuranceClaim.objects.create(
                patient=patient,
                insurance_policy=policy,
                service_date=today,
                diagnosis_codes="R42",  # Example ICD-10 code
                procedure_codes="92002",  # Example CPT code
                claim_amount=insurance_coverage,
                approval_status='pending'
            )
    except Exception as e:
        # Log error but continue with invoice generation
        print(f"Error processing insurance: {str(e)}")
        
    # Create the invoice
    due_date = timezone.now().date() + timedelta(days=30)
    
    invoice = Invoice.objects.create(
        patient=patient,
        amount=total_cost,
        tax=tax_amount,
        total_amount=total_cost + tax_amount,
        insurance_coverage=insurance_coverage,
        patient_responsibility=(total_cost + tax_amount) - insurance_coverage,
        due_date=due_date,
        notes=f"Invoice for prescription {dispensation.prescription.id}",
        medication_dispensation=dispensation
    )
    
    # Send notification
    from notification.services import send_billing_notification
    send_billing_notification(invoice)
    
    return invoice


def process_insurance_payment(claim):
    """Process payment from insurance for approved claims"""
    if claim.approval_status not in ['approved', 'partial']:
        raise ValueError("Can only process payment for approved or partially approved claims")
    
    if not claim.approved_amount:
        raise ValueError("Approved amount must be set")
    
    # Find related invoice
    try:
        invoice = Invoice.objects.get(insurance_claim=claim)
    except Invoice.DoesNotExist:
        # Try to find by patient and around the same date
        invoice = Invoice.objects.filter(
            patient=claim.patient,
            issue_date__gte=claim.service_date,
            issue_date__lte=claim.service_date + timedelta(days=7),
            status__in=['pending', 'partial', 'overdue']
        ).first()
        
        if not invoice:
            raise ValueError("No matching invoice found")
        
        # Link claim to invoice
        invoice.insurance_claim = claim
        invoice.save()
    
    # Create payment record
    from .models import Payment
    payment = Payment.objects.create(
        invoice=invoice,
        amount=claim.approved_amount,
        payment_method='insurance',
        transaction_id=f"INS-{claim.claim_number}",
        notes=f"Insurance payment for claim {claim.claim_number}"
    )
    
    # Update insurance coverage on invoice
    invoice.insurance_coverage = claim.approved_amount
    invoice.patient_responsibility = invoice.total_amount - claim.approved_amount
    invoice.save()
    
    # Send payment confirmation
    from notification.services import send_payment_confirmation
    send_payment_confirmation(payment)
    
    return payment
