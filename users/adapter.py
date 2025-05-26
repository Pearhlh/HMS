from django.conf import settings
from allauth.account.adapter import DefaultAccountAdapter

class CustomAccountAdapter(DefaultAccountAdapter):
    def populate_username(self, request, user):
        """
        Override this method to skip username population since our User model doesn't have a username field.
        """
        # Don't set a username - using email only
        pass

    def save_user(self, request, user, form, commit=True):
        """
        This is called when saving user via allauth registration.
        We override this to set custom fields on user model.
        """
        # In our model, we don't have username field, so get data without using parent method
        user.email = form.cleaned_data.get('email')
        user.set_password(form.cleaned_data.get('password1'))
        
        # Set custom fields from the form data
        user.first_name = form.cleaned_data.get('first_name', '')
        user.last_name = form.cleaned_data.get('last_name', '')
        user.role = form.cleaned_data.get('role', 'patient')
        
        if commit:
            user.save()
        return user
