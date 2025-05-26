import os
import django

# Initialize Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'healthcare.settings')
django.setup()

# Import User model after Django setup
from django.contrib.auth import get_user_model
from django.db.utils import IntegrityError

User = get_user_model()

# Get superuser credentials from environment variables or use defaults
DJANGO_SUPERUSER_USERNAME = os.environ.get('DJANGO_SUPERUSER_USERNAME', 'admin')
DJANGO_SUPERUSER_EMAIL = os.environ.get('DJANGO_SUPERUSER_EMAIL', 'admin@example.com')
DJANGO_SUPERUSER_PASSWORD = os.environ.get('DJANGO_SUPERUSER_PASSWORD', 'adminpassword')

try:
    # Try to create superuser
    User.objects.create_superuser(
        username=DJANGO_SUPERUSER_USERNAME,
        email=DJANGO_SUPERUSER_EMAIL,
        password=DJANGO_SUPERUSER_PASSWORD
    )
    print(f'Superuser "{DJANGO_SUPERUSER_USERNAME}" created successfully!')
except IntegrityError:
    print(f'Superuser "{DJANGO_SUPERUSER_USERNAME}" already exists.')
except Exception as e:
    print(f'Error creating superuser: {e}')
