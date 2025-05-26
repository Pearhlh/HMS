from django.urls import path
from . import views

urlpatterns = [
    path('management/', views.user_management, name='user_management'),
    path('list/', views.user_list, name='user_list'),
    path('create/', views.create_user, name='create_user'),
    path('<int:pk>/', views.user_detail, name='user_detail'),
    path('<int:pk>/update/', views.update_user, name='update_user'),
    path('<int:pk>/set-password/', views.set_user_password, name='set_user_password'),
    path('<int:pk>/toggle-status/', views.toggle_user_status, name='toggle_user_status'),
]
