from django.urls import path, include, re_path
from django.contrib import admin
from django.views.generic import TemplateView
from django.conf import settings
from django.conf.urls.static import static

# Add root URL pattern at the beginning of urlpatterns
urlpatterns = [
    # Root URL - home page
    path('', TemplateView.as_view(template_name='home.html'), name='home'),
    
    # Admin site
    path('admin/', admin.site.urls),
    
    # API routes
    path('api/debug/urls/', lambda request: None, name='debug_urls'),
    path('api/discovery/', lambda request: None, name='service-discovery'),
    path('api/health', lambda request: None, name='health_check'),
    path('api/health/', lambda request: None, name='health_check_slash'),
    path('api/<str:service>/health/', lambda request: None, name='service_health_slash'),
    path('api/<str:service>/health', lambda request: None, name='service_health'),
    re_path(r'^api/(?P<service>\w+)/(?P<path>.*)/$$', lambda request, service, path: None, name='api_proxy_slash'),
    re_path(r'^api/(?P<service>\w+)/(?P<path>.*)$$', lambda request, service, path: None, name='api_proxy'),
    path('api/mock/shipments/', lambda request: None, name='mock-shipments'),
    path('api/mock/shipments/<str:shipment_id>/', lambda request: None, name='mock-shipment-detail'),
    path('api/services/health/', lambda request: None, name='service-health-check'),
    
    # Add healthcare routes
    path('accounts/', include('allauth.urls')),
    path('patient/', include('patient.urls', namespace='patient')),
]

# Add static files handling
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
