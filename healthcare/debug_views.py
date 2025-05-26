from django.http import HttpResponse
from django.conf import settings
import os

def debug_settings(request):
    """Debug view to show current settings"""
    settings_info = [
        f"ROOT_URLCONF: {settings.ROOT_URLCONF}",
        f"SETTINGS_MODULE: {os.environ.get('DJANGO_SETTINGS_MODULE')}",
        f"BASE_DIR: {settings.BASE_DIR}",
        f"Current working directory: {os.getcwd()}"
    ]
    
    return HttpResponse("<br>".join(settings_info), content_type="text/html")
