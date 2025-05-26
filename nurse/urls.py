from django.urls import path
from . import views

app_name = 'nurse'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('patients/', views.patient_list, name='patient_list'),
    path('patient/<int:patient_id>/', views.patient_detail, name='patient_detail'),
    path('patient/<int:patient_id>/update-vitals/', views.update_vitals, name='update_vitals'),
]
