from django.core.management.base import BaseCommand
from appointment.models import Department

class Command(BaseCommand):
    help = 'Creates initial department data for the hospital'

    def handle(self, *args, **kwargs):
        departments = [
            {'name': 'Cardiology', 'description': 'Heart and cardiovascular system'},
            {'name': 'Neurology', 'description': 'Brain, spine, and nervous system'},
            {'name': 'Pediatrics', 'description': 'Child and adolescent health'},
            {'name': 'Orthopedics', 'description': 'Bones, joints, and muscles'},
            {'name': 'Dermatology', 'description': 'Skin conditions'},
            {'name': 'General Medicine', 'description': 'Overall health and primary care'},
            {'name': 'Gynecology', 'description': 'Female reproductive health'},
        ]
        
        for dept in departments:
            Department.objects.get_or_create(
                name=dept['name'],
                defaults={'description': dept['description']}
            )
        
        self.stdout.write(self.style.SUCCESS(f'Created {len(departments)} departments'))
