# Generated by Django 4.2.7 on 2025-04-15 05:12

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('patient', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='InsuranceProvider',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('address', models.TextField()),
                ('phone', models.CharField(max_length=20)),
                ('email', models.EmailField(max_length=254)),
                ('website', models.URLField(blank=True, null=True)),
                ('description', models.TextField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='InsurancePolicy',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('policy_number', models.CharField(max_length=50, unique=True)),
                ('member_id', models.CharField(max_length=50)),
                ('group_number', models.CharField(blank=True, max_length=50, null=True)),
                ('start_date', models.DateField()),
                ('end_date', models.DateField()),
                ('status', models.CharField(choices=[('active', 'Active'), ('expired', 'Expired'), ('canceled', 'Canceled'), ('pending', 'Pending Approval')], default='active', max_length=20)),
                ('coverage_percentage', models.IntegerField(help_text='Coverage percentage (0-100)')),
                ('coverage_details', models.TextField(help_text='Details about what is covered')),
                ('deductible', models.DecimalField(decimal_places=2, default=0, max_digits=10)),
                ('co_pay', models.DecimalField(decimal_places=2, default=0, max_digits=10)),
                ('out_of_pocket_max', models.DecimalField(decimal_places=2, default=0, max_digits=10)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='insurance_policies', to='patient.patient')),
                ('provider', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='policies', to='insurance.insuranceprovider')),
            ],
        ),
        migrations.CreateModel(
            name='InsuranceClaim',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('claim_number', models.CharField(max_length=50, unique=True)),
                ('service_date', models.DateField()),
                ('claim_date', models.DateField(auto_now_add=True)),
                ('claim_amount', models.DecimalField(decimal_places=2, max_digits=10)),
                ('approved_amount', models.DecimalField(blank=True, decimal_places=2, max_digits=10, null=True)),
                ('approval_status', models.CharField(choices=[('pending', 'Pending'), ('approved', 'Approved'), ('partial', 'Partially Approved'), ('rejected', 'Rejected'), ('canceled', 'Canceled')], default='pending', max_length=20)),
                ('diagnosis_codes', models.CharField(help_text='Comma-separated diagnosis codes', max_length=255)),
                ('service_codes', models.CharField(help_text='Comma-separated service codes', max_length=255)),
                ('notes', models.TextField(blank=True, null=True)),
                ('processed_date', models.DateTimeField(blank=True, null=True)),
                ('insurance_policy', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='claims', to='insurance.insurancepolicy')),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='insurance_claims', to='patient.patient')),
                ('processed_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='processed_claims', to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
