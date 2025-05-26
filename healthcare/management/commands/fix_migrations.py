import os
from django.core.management.base import BaseCommand
from django.db import connection

class Command(BaseCommand):
    help = 'Fix inconsistent migration history'

    def handle(self, *args, **options):
        self.stdout.write('Starting migration history fix...')
        
        with connection.cursor() as cursor:
            # Check if django_migrations table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='django_migrations';")
            if cursor.fetchone() is None:
                self.stdout.write(self.style.ERROR('django_migrations table not found.'))
                return
                
            # Backup existing migrations
            cursor.execute("SELECT app, name FROM django_migrations;")
            existing_migrations = cursor.fetchall()
            self.stdout.write(f'Found {len(existing_migrations)} migrations in database.')
            
            # Delete all records
            cursor.execute("DELETE FROM django_migrations;")
            self.stdout.write(self.style.SUCCESS('Migration history cleared.'))
        
        self.stdout.write(self.style.SUCCESS(
            'Migration history has been reset. Now run:\n'
            'python manage.py migrate --fake-initial'
        ))
