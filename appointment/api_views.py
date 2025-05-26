from django.http import JsonResponse
from .models import Doctor, Appointment
from django.utils import timezone
import datetime

def get_doctors_by_department(request, department_id):
    """API view to get doctors by department"""
    doctors = Doctor.objects.filter(department_id=department_id)
    doctor_list = []
    
    for doctor in doctors:
        doctor_list.append({
            'id': doctor.id,
            'name': f"Dr. {doctor.user.get_full_name()}",
            'specialization': doctor.specialization,
            'rating': doctor.rating if hasattr(doctor, 'rating') else None,
            'experience': doctor.experience_years if hasattr(doctor, 'experience_years') else None,
            'initials': f"{doctor.user.first_name[0]}{doctor.user.last_name[0]}",
        })
    
    return JsonResponse(doctor_list, safe=False)

def get_available_slots(request, doctor_id):
    """API view to get available time slots for a doctor on a specific date"""
    date_str = request.GET.get('date', '')
    
    if not date_str:
        return JsonResponse({'error': 'Date parameter is required'}, status=400)
    
    try:
        date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return JsonResponse({'error': 'Invalid date format. Use YYYY-MM-DD'}, status=400)
    
    doctor = Doctor.objects.filter(id=doctor_id).first()
    if not doctor:
        return JsonResponse({'error': 'Doctor not found'}, status=404)
    
    appointments = Appointment.objects.filter(doctor=doctor, date=date)
    booked_slots = [appointment.time for appointment in appointments]
    
    all_slots = [datetime.time(hour, 0) for hour in range(9, 17)]  # Example: 9 AM to 5 PM
    available_slots = [slot.strftime('%H:%M') for slot in all_slots if slot not in booked_slots]
    
    return JsonResponse({'available_slots': available_slots})