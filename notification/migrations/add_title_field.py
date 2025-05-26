from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies = [
        ('notification', '0001_initial'),  # Make sure this matches your last migration
    ]

    operations = [
        migrations.RenameField(
            model_name='notificationrecord',
            old_name='subject',  # The actual field name in the database
            new_name='title',    # The new field name we want to use
        ),
        # In case the field doesn't exist at all, add it
        migrations.AddField(
            model_name='notificationrecord',
            name='title',
            field=models.CharField(max_length=200, default='Notification'),
            preserve_default=False,
        ),
    ]
