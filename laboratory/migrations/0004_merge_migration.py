from django.db import migrations

class Migration(migrations.Migration):
    """
    This migration is created to merge conflicting migrations:
    - 0003_add_missing_fields
    - add_missing_columns
    """

    dependencies = [
        ('laboratory', '0003_add_missing_fields'),
        ('laboratory', 'add_missing_columns'),
    ]

    operations = [
        # No operations needed, this just merges branches
    ]
