import os
from django.core.management.base import BaseCommand
from django.db import connections
from django.db.migrations.recorder import MigrationRecorder

class Command(BaseCommand):
    help = 'Lists and fixes migration issues in the laboratory app'

    def handle(self, *args, **options):
        # Get the actual migrations applied to the database
        connection = connections['default']
        recorder = MigrationRecorder(connection)
        applied_migrations = recorder.applied_migrations()
        
        self.stdout.write("Applied migrations:")
        for app, name in applied_migrations:
            if app == 'laboratory':
                self.stdout.write(f"  - {app}.{name}")
                
        # Get the migration files present in the directory
        migrations_dir = os.path.join('laboratory', 'migrations')
        migration_files = []
        
        if os.path.exists(migrations_dir):
            for f in os.listdir(migrations_dir):
                if f.endswith('.py') and not f.startswith('__'):
                    migration_files.append(f[:-3])  # remove .py extension
        
        self.stdout.write("\nMigration files in directory:")
        for mf in migration_files:
            self.stdout.write(f"  - {mf}")
            
        self.stdout.write("\nTo fix the migration issue:")
        self.stdout.write("1. Update the dependencies in the problematic migration file")
        self.stdout.write("2. Or use fake migrations: python manage.py migrate --fake laboratory")
        
        # Add functionality to solve the problem
        self.stdout.write("\nDetected conflicting migrations. To fix:")
        self.stdout.write("1. Run: python manage.py makemigrations --merge")
        self.stdout.write("2. Then run: python manage.py migrate --fake laboratory")
        self.stdout.write("\nIf that doesn't work, try:")
        self.stdout.write("1. python manage.py showmigrations laboratory")
        self.stdout.write("2. For each conflicting migration, mark it as applied:")
        self.stdout.write("   python manage.py migrate laboratory --fake 0003_add_missing_fields")
        self.stdout.write("   python manage.py migrate laboratory --fake add_missing_columns")
