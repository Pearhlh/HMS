from django.urls import path
from . import views

urlpatterns = [
    path('', views.reports_dashboard, name='reports_dashboard'),
    path('list/', views.report_list, name='report_list'),
    path('config/create/', views.create_report_config, name='create_report_config'),
    path('config/<int:pk>/parameters/', views.report_parameters, name='report_parameters'),
    path('view/<int:pk>/', views.view_report, name='view_report'),
]
