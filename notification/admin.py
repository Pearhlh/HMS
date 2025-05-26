from django.contrib import admin
from .models import NotificationRecord

@admin.register(NotificationRecord)
class NotificationRecordAdmin(admin.ModelAdmin):
    # Fix field names to match database schema
    list_display = ['subject', 'user', 'notification_type', 'created_at', 'read']
    list_filter = ['notification_type', 'read', 'created_at']
    search_fields = ['subject', 'message', 'user__username', 'user__first_name', 'user__last_name']
    date_hierarchy = 'created_at'
