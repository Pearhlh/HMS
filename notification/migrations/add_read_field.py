from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        ('notification', '0001_initial'),  # Adjust this to match your last migration
    ]

    operations = [
        migrations.AddField(
            model_name='notificationrecord',
            name='read',
            field=models.BooleanField(default=False),
        ),
    ]
