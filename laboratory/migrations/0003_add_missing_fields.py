from django.db import migrations, models
import django.db.models.deletion

class Migration(migrations.Migration):

    dependencies = [
        ('laboratory', '0001_initial'),  # Update dependency to reference existing migration
    ]

    operations = [
        migrations.AddField(
            model_name='labtest',
            name='test_code',
            field=models.CharField(blank=True, max_length=50, null=True, unique=True),
        ),
        migrations.AddField(
            model_name='labtest',
            name='test_name',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AddField(
            model_name='labtest',
            name='test_date',
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='labtest',
            name='sample_type',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
    ]
