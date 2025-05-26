def get_lab_tests_safely(patient_id):
    """
    Retrieves lab tests while handling missing columns gracefully
    """
    from django.db import connection
    
    # First check what columns actually exist
    safe_fields = ['id', 'patient_id', 'test_type', 'status', 'requested_date', 'description']
    
    # Query to get lab tests with only the columns that exist
    query = f"""
    SELECT id, patient_id, test_type, status, 
           requested_date, description
    FROM laboratory_labtest 
    WHERE patient_id = %s
    ORDER BY requested_date DESC
    """
    
    with connection.cursor() as cursor:
        try:
            cursor.execute(query, [patient_id])
            columns = [col[0] for col in cursor.description]
            
            lab_tests = []
            for row in cursor.fetchall():
                test = dict(zip(columns, row))
                # Add display methods
                test['get_status_display'] = lambda status=test['status']: {
                    'requested': 'Requested',
                    'scheduled': 'Scheduled',
                    'in_progress': 'In Progress',
                    'completed': 'Completed',
                    'canceled': 'Canceled'
                }.get(status, status.title() if status else '')
                
                test['get_test_type_display'] = lambda t_type=test['test_type']: {
                    'blood': 'Blood Test',
                    'urine': 'Urine Test',
                    'imaging': 'Imaging',
                    'general': 'General',
                    'other': 'Other'
                }.get(t_type, t_type.title() if t_type else '')
                
                lab_tests.append(test)
            
            return lab_tests
        except Exception as e:
            print(f"Error retrieving lab tests: {e}")
            return []
