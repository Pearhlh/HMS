from django.contrib import admin
from django.urls import path, include, re_path
from django.views.generic import TemplateView
from django.conf import settings
from django.conf.urls.static import static
from .views import login_redirect
from appointment import api_views as appointment_api_views
from .debug_views import debug_settings  # Import the debug view

# Import drf-yasg components
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

# API Documentation info
api_info = openapi.Info(
    title="Healthcare System API",
    default_version='v1',
    description="API documentation for the Healthcare System",
    terms_of_service="https://www.example.com/terms/",
    contact=openapi.Contact(email="contact@example.com"),
    license=openapi.License(name="BSD License"),
)

# Create schema view for Swagger documentation
schema_view = get_schema_view(
    api_info,
    public=True,
    permission_classes=[permissions.AllowAny],
)

urlpatterns = [
    # Debug route to check settings
    path('debug/settings/', debug_settings, name='debug_settings'),
    
    # API documentation routes
    path('api/docs/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('api/redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
    
    # Include API endpoints
    path('api/', include('api.urls')),
    
    # Root URL - home page
    path('', TemplateView.as_view(template_name='home.html'), name='home'),
    
    path('admin/', admin.site.urls),
    # Authentication
    path('accounts/', include('allauth.urls')),
    
    # Patient routes
    path('patient/', include('patient.urls', namespace='patient')),
    
    # Doctor routes
    path('doctor/', include('doctor.urls', namespace='doctor')),
    
    # Laboratory routes
    path('laboratory/', include('laboratory.urls', namespace='laboratory')),
    
    # Pharmacy routes
    path('pharmacy/', include('pharmacy.urls', namespace='pharmacy')),
    
    # Nurse routes
    path('nurse/', include('nurse.urls', namespace='nurse')),
    
    # Insurance routes
    path('insurance/', include('insurance.urls', namespace='insurance')),
    
    # Login redirect
    path('login-redirect/', login_redirect, name='login_redirect'),
    
    # API endpoints for appointments
    path('api/departments/<int:department_id>/doctors/', 
         appointment_api_views.get_doctors_by_department, 
         name='api_doctors_by_department'),
    path('api/doctors/<int:doctor_id>/available-slots/', 
         appointment_api_views.get_available_slots, 
         name='api_doctor_available_slots'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
