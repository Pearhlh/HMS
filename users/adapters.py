from allauth.account.adapter import DefaultAccountAdapter
from django.conf import settings
from django.urls import reverse

class CustomAccountAdapter(DefaultAccountAdapter):
    def save_user(self, request, user, form, commit=True):
        """
        Override to set custom user fields during registration
        """
        user = super().save_user(request, user, form, commit=False)
        
        # Set default role to patient for new signups
        user.role = 'patient'
        
        # Get any additional fields from the form
        data = form.cleaned_data
        
        # Save first name and last name if provided
        if 'first_name' in data:
            user.first_name = data['first_name']
        if 'last_name' in data:
            user.last_name = data['last_name']
        
        if commit:
            user.save()
            
        return user
    
    def get_signup_redirect_url(self, request):
        """
        Redirect new users to the patient profile creation page
        """
        return reverse('patient_profile')
