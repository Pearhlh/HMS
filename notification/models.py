from django.db import models
from django.conf import settings
from django.utils import timezone

class NotificationRecord(models.Model):
    """Model to store user notifications"""
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='notifications')
    subject = models.CharField(max_length=255)
    message = models.TextField()
    notification_type = models.CharField(max_length=50, default='general')
    created_at = models.DateTimeField(default=timezone.now)
    read = models.BooleanField(default=False)
    action_url = models.CharField(max_length=255, blank=True, null=True)
    action_text = models.CharField(max_length=50, blank=True, null=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.subject} - {self.user.username}"
