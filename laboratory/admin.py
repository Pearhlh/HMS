from django.contrib import admin
from .models import LabTest, LabResultItem, TestResult

class LabResultItemInline(admin.TabularInline):
    model = LabResultItem
    extra = 1

@admin.register(LabTest)
class LabTestAdmin(admin.ModelAdmin):
    list_display = ['test_name', 'patient', 'requested_by', 'status', 'requested_date', 'completed_date']
    list_filter = ['status', 'test_type', 'priority']
    search_fields = ['test_name', 'patient__user__first_name', 'patient__user__last_name']
    inlines = [LabResultItemInline]

@admin.register(LabResultItem)
class LabResultItemAdmin(admin.ModelAdmin):
    list_display = ['parameter_name', 'result', 'unit', 'is_abnormal']
    list_filter = ['is_abnormal', 'test']
    search_fields = ['parameter_name', 'test__test_name']

@admin.register(TestResult)
class TestResultAdmin(admin.ModelAdmin):
    list_display = ['lab_test', 'result_value', 'is_abnormal', 'uploaded_at']
    list_filter = ['is_abnormal', 'uploaded_at']
    search_fields = ['lab_test__test_name']
