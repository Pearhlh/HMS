import uuid
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.views.generic import ListView, DetailView, CreateView
from django.urls import reverse_lazy
from django.contrib import messages
from django.db.models import Q, Sum
from django.http import JsonResponse
from django.utils import timezone
from datetime import timedelta
from .models import Invoice, Payment
from .forms import InvoiceForm, PaymentForm
from patient.models import Patient
from doctor.models import Doctor
from appointment.models import Appointment
from insurance.models import InsuranceClaim, InsurancePolicy

@login_required
def billing_dashboard(request):
    """Dashboard for billing management"""
    if not (request.user.is_admin or request.user.is_superuser):
        messages.error(request, "You don't have access to the billing dashboard.")
        return redirect('home')
    
    # Get recent invoices
    recent_invoices = Invoice.objects.all().order_by('-issue_date')[:10]
    
    # Get overdue invoices
    overdue_invoices = Invoice.objects.filter(
        status='overdue'
    ).order_by('due_date')
    
    # Get payment statistics
    today = timezone.now().date()
    start_of_month = today.replace(day=1)
    
    payments_today = Payment.objects.filter(
        payment_date__date=today
    ).aggregate(total=Sum('amount'))['total'] or 0
    
    payments_this_month = Payment.objects.filter(
        payment_date__date__gte=start_of_month
    ).aggregate(total=Sum('amount'))['total'] or 0
    
    # Get invoice statistics
    pending_amount = Invoice.objects.filter(
        status__in=['pending', 'partial', 'overdue']
    ).aggregate(total=Sum('patient_responsibility'))['total'] or 0
    
    total_invoiced = Invoice.objects.aggregate(total=Sum('total_amount'))['total'] or 0
    total_paid = Payment.objects.aggregate(total=Sum('amount'))['total'] or 0
    
    context = {
        'recent_invoices': recent_invoices,
        'overdue_invoices': overdue_invoices,
        'payments_today': payments_today,
        'payments_this_month': payments_this_month,
        'pending_amount': pending_amount,
        'total_invoiced': total_invoiced,
        'total_paid': total_paid,
    }
    return render(request, 'billing/dashboard.html', context)


class InvoiceListView(LoginRequiredMixin, ListView):
    """View for listing invoices"""
    model = Invoice
    template_name = 'billing/invoice_list.html'
    context_object_name = 'invoices'
    paginate_by = 10
    
    def get_queryset(self):
        queryset = Invoice.objects.all().order_by('-issue_date')
        user = self.request.user
        
        # Filter by user role
        if user.is_patient:
            try:
                queryset = queryset.filter(patient__user=user)
            except:
                queryset = Invoice.objects.none()
        
        # Filter by status if provided
        status = self.request.GET.get('status')
        if status:
            queryset = queryset.filter(status=status)
            
        # Filter by search term
        search = self.request.GET.get('search')
        if search:
            queryset = queryset.filter(
                Q(invoice_number__icontains=search) |
                Q(patient__user__first_name__icontains=search) |
                Q(patient__user__last_name__icontains=search)
            )
            
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['status_filter'] = self.request.GET.get('status', '')
        context['search_term'] = self.request.GET.get('search', '')
        context['status_choices'] = Invoice.STATUS_CHOICES
        return context


@login_required
def create_invoice(request):
    """View for creating a new invoice"""
    if not (request.user.is_admin or request.user.is_superuser):
        messages.error(request, "You don't have permission to create invoices.")
        return redirect('invoice_list')
    
    if request.method == 'POST':
        form = InvoiceForm(request.POST)
        
        if form.is_valid():
            invoice = form.save(commit=False)
            invoice.created_by = request.user
            invoice.save()
            messages.success(request, f"Invoice {invoice.invoice_number} created successfully")
            return redirect('invoice_detail', pk=invoice.id)
    else:
        form = InvoiceForm()
    
    return render(request, 'billing/create_invoice.html', {
        'form': form,
    })


@login_required
def invoice_detail(request, pk):
    """View for invoice details"""
    invoice = get_object_or_404(Invoice, pk=pk)
    
    # Check permissions
    user = request.user
    if user.is_patient and invoice.patient.user != user:
        messages.error(request, "You don't have permission to view this invoice.")
        return redirect('invoice_list')
    
    # Get related payments
    payments = Payment.objects.filter(invoice=invoice).order_by('-payment_date')
    
    return render(request, 'billing/invoice_detail.html', {
        'invoice': invoice,
        'payments': payments,
    })


@login_required
def record_payment(request, invoice_id):
    """View for recording a payment for an invoice"""
    invoice = get_object_or_404(Invoice, pk=invoice_id)
    
    # Check permissions
    if not (request.user.is_admin or request.user.is_superuser):
        messages.error(request, "You don't have permission to record payments.")
        return redirect('invoice_detail', pk=invoice.id)
    
    if request.method == 'POST':
        form = PaymentForm(request.POST)
        
        if form.is_valid():
            payment = form.save(commit=False)
            payment.invoice = invoice
            payment.processed_by = request.user
            payment.save()
            
            messages.success(request, f"Payment of ${payment.amount} recorded successfully")
            return redirect('invoice_detail', pk=invoice.id)
    else:
        # Initialize with remaining amount
        form = PaymentForm(initial={'amount': invoice.amount_due})
    
    return render(request, 'billing/record_payment.html', {
        'form': form,
        'invoice': invoice,
    })


@login_required
def payment_confirmation(request, payment_id):
    """View for payment confirmation"""
    payment = get_object_or_404(Payment, pk=payment_id)
    invoice = payment.invoice
    
    # Check permissions
    user = request.user
    if user.is_patient and invoice.patient.user != user:
        messages.error(request, "You don't have permission to view this payment.")
        return redirect('invoice_list')
    
    return render(request, 'billing/payment_confirmation.html', {
        'payment': payment,
        'invoice': invoice,
    })


@login_required
def process_online_payment(request, invoice_id):
    """View for processing online payments"""
    invoice = get_object_or_404(Invoice, pk=invoice_id)
    
    # Check permissions
    user = request.user
    if user.is_patient and invoice.patient.user != user:
        messages.error(request, "You don't have permission to pay this invoice.")
        return redirect('invoice_list')
    
    if request.method == 'POST':
        # This would integrate with a payment gateway in a real application
        # For demo purposes, we'll just create a payment record
        
        payment = Payment.objects.create(
            invoice=invoice,
            amount=float(request.POST.get('amount', invoice.amount_due)),
            payment_method='online',
            transaction_id=f"TRANS-{uuid.uuid4().hex[:8].upper()}",
            processed_by=request.user
        )
        
        messages.success(request, "Payment processed successfully!")
        return redirect('payment_confirmation', payment_id=payment.id)
    
    return render(request, 'billing/process_payment.html', {
        'invoice': invoice,
    })


@login_required
def invoice_list(request):
    """View for listing invoices"""
    # Placeholder implementation
    return render(request, 'billing/invoice_list.html', {})

@login_required
def process_payment(request, pk):
    """View for processing payments"""
    # Placeholder implementation
    return render(request, 'billing/process_payment.html', {})

@login_required
def financial_reports(request):
    """View for financial reports"""
    # Placeholder implementation
    return render(request, 'billing/financial_reports.html', {})
