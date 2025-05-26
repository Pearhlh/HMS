from django.db import migrations

class Migration(migrations.Migration):

    dependencies = [
        ('laboratory', '0001_initial'),  # Make sure this matches your last migration
    ]

    operations = [
        migrations.RunSQL(
            sql="""
            ALTER TABLE laboratory_labtest ADD COLUMN IF NOT EXISTS test_code VARCHAR(50) NULL;
            ALTER TABLE laboratory_labtest ADD COLUMN IF NOT EXISTS test_name VARCHAR(100) NULL;
            ALTER TABLE laboratory_labtest ADD COLUMN IF NOT EXISTS test_date DATETIME NULL;
            ALTER TABLE laboratory_labtest ADD COLUMN IF NOT EXISTS sample_type VARCHAR(50) NULL;
            """,
            reverse_sql="""
            -- No reverse operation needed
            """
        ),
    ]
