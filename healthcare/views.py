from django.shortcuts import redirect
from django.urls import reverse
from django.contrib import messages

def login_redirect(request):
    """Redirect logged-in users to their appropriate dashboard based on role"""
    user = request.user
    
    if not user.is_authenticated:
        return redirect('account_login')
        
    try:
        if user.role == 'patient':
            return redirect('patient:dashboard')
        elif user.role == 'doctor':
            try:
                return redirect('doctor:dashboard')
            except:
                messages.info(request, "Doctor dashboard is not available yet. Redirecting to home.")
                return redirect('home')
        elif user.role == 'nurse':
            try:
                return redirect('nurse:dashboard')
            except Exception as e:
                messages.error(request, f"Error accessing nurse dashboard: {str(e)}. Redirecting to home.")
                return redirect('home')
        elif user.role == 'lab_tech':
            try:
                return redirect('laboratory:dashboard')
            except:
                messages.info(request, "Laboratory dashboard is not available yet. Redirecting to home.")
                return redirect('home')
        elif user.role in ['pharmacy', 'pharmacist']:
            try:
                return redirect('pharmacy:dashboard')
            except:
                messages.info(request, "Pharmacy dashboard is not available yet. Redirecting to home.")
                return redirect('home')
        elif user.role == 'admin' or user.is_staff:
            return redirect('admin:index')
        elif user.role in ['insurance', 'insurer']:
            try:
                return redirect('insurance:dashboard')
            except Exception as e:
                messages.error(request, f"Error accessing insurance dashboard: {str(e)}. Redirecting to home.")
                return redirect('home')
        else:
            messages.warning(request, f"No dashboard available for role: {user.role}. Redirecting to home.")
            return redirect('home')
    except Exception as e:
        # Fallback to home page if any issues occur
        messages.error(request, f"Error during redirection: {str(e)}")
        return redirect('home')
