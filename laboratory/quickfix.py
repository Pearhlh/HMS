from django.db import connection

def fix_lab_test_columns():
    """
    Quick fix to add missing columns to the laboratory_labtest table
    Run this from a Django shell (python manage.py shell)
    """
    with connection.cursor() as cursor:
        # Check if columns exist and add them if they don't
        try:
            # For SQLite - using separate statements for better compatibility
            cursor.execute("PRAGMA table_info(laboratory_labtest)")
            existing_columns = [column[1] for column in cursor.fetchall()]
            
            if 'test_code' not in existing_columns:
                cursor.execute("ALTER TABLE laboratory_labtest ADD COLUMN test_code VARCHAR(50) NULL")
                print("Added test_code column")
                
            if 'test_name' not in existing_columns:
                cursor.execute("ALTER TABLE laboratory_labtest ADD COLUMN test_name VARCHAR(100) NULL")
                print("Added test_name column")
                
            if 'test_date' not in existing_columns:
                cursor.execute("ALTER TABLE laboratory_labtest ADD COLUMN test_date DATETIME NULL")
                print("Added test_date column")
                
            if 'sample_type' not in existing_columns:
                cursor.execute("ALTER TABLE laboratory_labtest ADD COLUMN sample_type VARCHAR(50) NULL")
                print("Added sample_type column")
                
            # Add foreign key columns
            if 'technician_id' not in existing_columns:
                cursor.execute("ALTER TABLE laboratory_labtest ADD COLUMN technician_id INTEGER NULL")
                print("Added technician_id column")
                
            if 'ordered_by_id' not in existing_columns:
                cursor.execute("ALTER TABLE laboratory_labtest ADD COLUMN ordered_by_id INTEGER NULL")
                print("Added ordered_by_id column")
            
            # Add priority column
            if 'priority' not in existing_columns:
                cursor.execute("ALTER TABLE laboratory_labtest ADD COLUMN priority VARCHAR(20) DEFAULT 'normal'")
                print("Added priority column")
                
            # Add notes column
            if 'notes' not in existing_columns:
                cursor.execute("ALTER TABLE laboratory_labtest ADD COLUMN notes TEXT NULL")
                print("Added notes column")
                
            print("Database columns check completed")
        except Exception as e:
            print(f"Error fixing columns: {e}")
            
        # Make sure duplicated name/test_name situation is handled
        try:
            # Check if name column exists first
            cursor.execute("PRAGMA table_info(laboratory_labtest)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'name' in columns and 'test_name' in columns:
                cursor.execute("SELECT id, name, test_name FROM laboratory_labtest")
                rows = cursor.fetchall()
                for row_id, name, test_name in rows:
                    if test_name is None and name:
                        cursor.execute(
                            "UPDATE laboratory_labtest SET test_name = ? WHERE id = ? AND test_name IS NULL", 
                            [name, row_id]
                        )
                print("Data consistency check completed")
            else:
                print("Name or test_name column might be missing - skipping data transfer")
        except Exception as e:
            print(f"Error updating data: {e}")

        # Check if LabResultItem table exists, create if not
        try:
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='laboratory_labresultitem'
            """)
            if not cursor.fetchone():
                print("Creating laboratory_labresultitem table...")
                cursor.execute("""
                    CREATE TABLE laboratory_labresultitem (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        test_id INTEGER NOT NULL REFERENCES laboratory_labtest(id),
                        result VARCHAR(100) NOT NULL,
                        normal_range VARCHAR(100) NULL,
                        is_abnormal BOOLEAN NOT NULL DEFAULT 0,
                        notes TEXT NULL,
                        parameter_name VARCHAR(100) NOT NULL,
                        value VARCHAR(100) NOT NULL,
                        unit VARCHAR(50) NULL,
                        reference_range VARCHAR(100) NULL,
                        FOREIGN KEY (test_id) REFERENCES laboratory_labtest (id)
                    )
                """)
                print("Created laboratory_labresultitem table")
            else:
                print("laboratory_labresultitem table already exists")
        except Exception as e:
            print(f"Error checking/creating LabResultItem table: {e}")

if __name__ == "__main__":
    print("\n*** DJANGO DATABASE QUICK FIX ***")
    print("This script must be run from the Django shell. Follow these steps:")
    print("1. Run: python manage.py shell")
    print("2. In the shell, type:")
    print("   from laboratory.quickfix import fix_lab_test_columns")
    print("   fix_lab_test_columns()")
    print("\nIf you're seeing this message, you're not running this in the Django shell.")
