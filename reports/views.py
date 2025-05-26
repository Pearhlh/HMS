from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import HttpResponse, JsonResponse
from django.contrib import messages
from django.utils import timezone
from django.db.models import Q
from datetime import datetime, timedelta
import csv
import json
import io

# Handle the missing xlsxwriter dependency
try:
    import xlsxwriter
    EXCEL_EXPORT_AVAILABLE = True
except ImportError:
    EXCEL_EXPORT_AVAILABLE = False

from .models import ReportConfiguration, GeneratedReport
from .forms import ReportConfigForm, ReportParametersForm

def is_staff_or_admin(user):
    """Check if user has permission to access reports"""
    return user.is_staff or user.is_admin or user.is_superuser

@login_required
def reports_dashboard(request):
    """Dashboard view showing available reports"""
    recent_reports = GeneratedReport.objects.filter(
        created_by=request.user
    ).order_by('-created_at')[:5]
    
    # Fix the syntax error - use Q objects properly with filter() method
    report_configurations = ReportConfiguration.objects.filter(
        Q(created_by=request.user) | Q(is_public=True)
    ).order_by('report_type', 'name')
    
    context = {
        'recent_reports': recent_reports,
        'report_configurations': report_configurations,
    }
    return render(request, 'reports/dashboard.html', context)

@login_required
def report_list(request):
    """View for listing all report configurations"""
    report_type = request.GET.get('type')
    
    if report_type:
        # Fix the syntax error - use Q objects properly with filter() method
        configurations = ReportConfiguration.objects.filter(
            report_type=report_type
        ).filter(
            Q(created_by=request.user) | Q(is_public=True)
        ).order_by('name')
    else:
        # Fix the syntax error - use Q objects properly with filter() method
        configurations = ReportConfiguration.objects.filter(
            Q(created_by=request.user) | Q(is_public=True)
        ).order_by('report_type', 'name')
    
    generated_reports = GeneratedReport.objects.filter(
        created_by=request.user
    ).order_by('-created_at')[:20]
    
    context = {
        'configurations': configurations,
        'generated_reports': generated_reports,
        'report_type': report_type,
    }
    return render(request, 'reports/report_list.html', context)

@login_required
@user_passes_test(is_staff_or_admin)
def create_report_config(request):
    """View for creating a new report configuration"""
    if request.method == 'POST':
        form = ReportConfigForm(request.POST)
        if form.is_valid():
            config = form.save(commit=False)
            config.created_by = request.user
            config.save()
            messages.success(request, f"Report configuration '{config.name}' created successfully")
            return redirect('report_parameters', pk=config.id)
    else:
        form = ReportConfigForm()
    
    return render(request, 'reports/create_config.html', {'form': form})

@login_required
def report_parameters(request, pk):
    """View for configuring report parameters and generating a report"""
    config = get_object_or_404(ReportConfiguration, pk=pk)
    
    # Check if user has access to this configuration
    if not (config.is_public or config.created_by == request.user or request.user.is_superuser):
        messages.error(request, "You don't have permission to access this report configuration")
        return redirect('report_list')
    
    if request.method == 'POST':
        form = ReportParametersForm(request.POST, report_type=config.report_type)
        if form.is_valid():
            # Create a new generated report
            report = GeneratedReport(
                configuration=config,
                name=f"{config.name} - {timezone.now().strftime('%Y-%m-%d %H:%M')}",
                parameters_used=form.cleaned_data,
                created_by=request.user
            )
            report.save()
            
            # Queue report generation (in a real system, this would use Celery or similar)
            # For now, we'll generate it immediately
            try:
                # This would call a method to actually generate the report based on the type
                if config.report_type == 'financial':
                    result_data = generate_financial_report(form.cleaned_data)
                elif config.report_type == 'patient':
                    result_data = generate_patient_report(form.cleaned_data)
                # ... other report types
                else:
                    result_data = {'message': 'Report type not implemented'}
                
                report.result_data = result_data
                report.status = 'completed'
                report.completed_at = timezone.now()
                report.save()
                
                messages.success(request, f"Report '{report.name}' generated successfully")
                return redirect('view_report', pk=report.id)
                
            except Exception as e:
                report.status = 'failed'
                report.result_data = {'error': str(e)}
                report.completed_at = timezone.now()
                report.save()
                messages.error(request, f"Failed to generate report: {str(e)}")
    else:
        # Create form with default parameters from configuration
        initial_data = config.parameters if hasattr(config, 'parameters') else {}
        form = ReportParametersForm(initial=initial_data, report_type=config.report_type)
    
    context = {
        'config': config,
        'form': form,
    }
    return render(request, 'reports/parameters.html', context)

@login_required
def view_report(request, pk):
    """View for displaying a generated report"""
    report = get_object_or_404(GeneratedReport, pk=pk)
    
    # Check if user has access to this report
    if report.created_by != request.user and not request.user.is_superuser:
        messages.error(request, "You don't have permission to view this report")
        return redirect('report_list')
    
    # Check if the report is ready
    if report.status != 'completed':
        messages.info(request, f"Report is {report.get_status_display()}")
        return redirect('report_list')
    
    # Handle export options
    export_format = request.GET.get('export')
    if export_format == 'csv':
        return export_report_csv(report)
    elif export_format == 'excel':
        return export_report_excel(report)
    elif export_format == 'pdf':
        messages.info(request, "PDF export not yet implemented")
    
    context = {
        'report': report,
    }
    return render(request, 'reports/view_report.html', context)

def export_report_csv(report):
    """Export report data as CSV"""
    if not report.result_data or 'data' not in report.result_data:
        return HttpResponse("No data to export", content_type="text/plain")
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write headers
    headers = report.result_data.get('headers', [])
    if headers:
        writer.writerow(headers)
    
    # Write data rows
    for row in report.result_data.get('data', []):
        writer.writerow(row)
    
    response = HttpResponse(output.getvalue(), content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{report.name}.csv"'
    return response

def export_report_excel(report):
    """Export report data as Excel"""
    if not report.result_data or 'data' not in report.result_data:
        return HttpResponse("No data to export", content_type="text/plain")
    
    # Check if xlsxwriter is available
    if not EXCEL_EXPORT_AVAILABLE:
        return HttpResponse("Excel export not available. Please install xlsxwriter package.", 
                           content_type="text/plain")
    
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output)
    worksheet = workbook.add_worksheet()
    
    # Write headers with formatting
    headers = report.result_data.get('headers', [])
    header_format = workbook.add_format({'bold': True, 'bg_color': '#D8E4BC'})
    for col, header in enumerate(headers):
        worksheet.write(0, col, header, header_format)
    
    # Write data rows
    for row_idx, row_data in enumerate(report.result_data.get('data', []), start=1):
        for col_idx, cell_value in enumerate(row_data):
            worksheet.write(row_idx, col_idx, cell_value)
    
    workbook.close()
    
    response = HttpResponse(output.getvalue(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = f'attachment; filename="{report.name}.xlsx"'
    return response

# Example report generation functions (these would be in separate modules in a real system)
def generate_financial_report(parameters):
    """Generate financial report based on parameters"""
    from_date = parameters.get('from_date')
    to_date = parameters.get('to_date')
    
    # In a real system, this would query the database
    # For this example, we'll return mock data
    headers = ['Date', 'Service', 'Patient', 'Amount', 'Insurance', 'Patient Responsibility', 'Status']
    data = [
        ['2023-10-01', 'Consultation', 'John Doe', '$150.00', '$120.00', '$30.00', 'Paid'],
        ['2023-10-02', 'Laboratory Test', 'Jane Smith', '$250.00', '$200.00', '$50.00', 'Pending'],
        ['2023-10-03', 'Prescription', 'Robert Brown', '$75.00', '$60.00', '$15.00', 'Paid'],
    ]
    
    # Calculate totals
    total_amount = sum(float(row[3].replace('$', '')) for row in data)
    total_insurance = sum(float(row[4].replace('$', '')) for row in data)
    total_patient = sum(float(row[5].replace('$', '')) for row in data)
    
    return {
        'headers': headers,
        'data': data,
        'summary': {
            'total_amount': f'${total_amount:.2f}',
            'total_insurance': f'${total_insurance:.2f}',
            'total_patient': f'${total_patient:.2f}',
            'total_records': len(data)
        }
    }

def generate_patient_report(parameters):
    """Generate patient statistics report"""
    # Mock data
    headers = ['Age Group', 'Male', 'Female', 'Other', 'Total']
    data = [
        ['0-18', 120, 105, 2, 227],
        ['19-35', 185, 210, 5, 400],
        ['36-50', 210, 230, 3, 443],
        ['51-65', 175, 185, 1, 361],
        ['65+', 145, 160, 0, 305]
    ]
    
    return {
        'headers': headers,
        'data': data,
        'summary': {
            'total_patients': sum(row[4] for row in data),
            'male_total': sum(row[1] for row in data),
            'female_total': sum(row[2] for row in data),
            'other_total': sum(row[3] for row in data)
        }
    }
