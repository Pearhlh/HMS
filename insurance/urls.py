from django.urls import path
from . import views

app_name = 'insurance'

urlpatterns = [
    path('dashboard/', views.insurance_dashboard, name='dashboard'),
    path('policies/', views.policy_list, name='policy_list'),
    path('policies/<int:pk>/', views.policy_detail, name='policy_detail'),
    # Commenting out the line causing the error - view doesn't exist
    # path('policies/create/', views.create_policy, name='create_policy'),
    path('claims/', views.InsuranceClaimListView.as_view(), name='claim_list'),
    path('claims/<int:pk>/', views.claim_detail, name='claim_detail'),
    path('claims/create/', views.create_claim, name='create_claim'),
    path('claims/create/<int:policy_id>/', views.create_claim, name='create_claim'),
    path('claims/<int:pk>/process/', views.process_claim, name='process_claim'),
    path('verification-requests/', views.verification_request_list, name='verification_requests'),
]

# Make sure other critical URLs are present
if 'policy_list' not in [url.name for url in urlpatterns]:
    urlpatterns.append(path('policies/', views.policy_list, name='policy_list'))
