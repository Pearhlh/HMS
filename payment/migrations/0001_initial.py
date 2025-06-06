# Generated by Django 4.2.7 on 2025-04-14 16:24

from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('patient', '0001_initial'),
        ('pharmacy', '0002_prescriptioninvoice_medicationdelivery'),
    ]

    operations = [
        migrations.CreateModel(
            name='Payment',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('amount', models.DecimalField(decimal_places=2, max_digits=10)),
                ('payment_method', models.CharField(choices=[('online', 'Online Payment'), ('credit_card', 'Credit Card'), ('debit_card', 'Debit Card'), ('cash', 'Cash'), ('insurance', 'Insurance')], max_length=20)),
                ('payment_date', models.DateTimeField(default=django.utils.timezone.now)),
                ('transaction_id', models.CharField(blank=True, max_length=100)),
                ('status', models.CharField(default='completed', max_length=20)),
                ('notes', models.TextField(blank=True)),
                ('invoice', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='payments', to='pharmacy.prescriptioninvoice')),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='payment_records', to='patient.patient')),
            ],
        ),
    ]
