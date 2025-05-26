from django.contrib import admin
from .models import ReportConfiguration, GeneratedReport

@admin.register(ReportConfiguration)
class ReportConfigurationAdmin(admin.ModelAdmin):
    list_display = ('name', 'report_type', 'created_by', 'is_public', 'created_at')
    list_filter = ('report_type', 'is_public', 'created_at')
    search_fields = ('name', 'description', 'created_by__email')
    date_hierarchy = 'created_at'

@admin.register(GeneratedReport)
class GeneratedReportAdmin(admin.ModelAdmin):
    list_display = ('name', 'configuration', 'status', 'created_by', 'created_at', 'completed_at')
    list_filter = ('status', 'created_at')
    search_fields = ('name', 'created_by__email', 'configuration__name')
    date_hierarchy = 'created_at'
    readonly_fields = ('created_at', 'completed_at', 'status', 'parameters_used', 'result_data')
