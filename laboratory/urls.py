from django.urls import path
from . import views

app_name = 'laboratory'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('tests/', views.test_list, name='test_list'),
    path('test/<int:test_id>/', views.test_detail, name='test_detail'),
    path('test/<int:test_id>/record-results/', views.record_results, name='record_results'),
    path('test/<int:test_id>/update-status/', views.update_test_status, name='update_test_status'),
    path('test/<int:test_id>/finalize/', views.finalize_lab_result, name='finalize_lab_result'),
]
