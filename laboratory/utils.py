from django.db import connection

def get_tests_safely(status=None):
    """
    Retrieves lab tests while handling missing columns gracefully
    
    Args:
        status (str, optional): Filter tests by status
    
    Returns:
        list: List of dictionaries containing test data
    """
    # Print debug info to see what status is being requested
    print(f"Fetching tests with status filter: '{status}'")
    
    # Build query dynamically
    query = """
    SELECT lt.id, lt.patient_id, lt.test_type, lt.status, 
           lt.requested_date, lt.description, COALESCE(lt.priority, 'normal') as priority
    FROM laboratory_labtest lt
    """
    
    params = []
    if status:
        query += " WHERE lt.status = ?"
        params.append(status)
        
    query += " ORDER BY lt.requested_date DESC"
    
    # Print the query for debugging
    print(f"Executing query: {query}")
    print(f"With parameters: {params}")
    
    with connection.cursor() as cursor:
        try:
            cursor.execute(query, params)
            columns = [col[0] for col in cursor.description]
            
            # Debug: Count the results
            results = cursor.fetchall()
            print(f"Query returned {len(results)} tests")
            
            tests = []
            for row in results:
                test = dict(zip(columns, row))
                
                # Debug: Print each test's status
                print(f"Test ID: {test['id']}, Status: {test['status']}")
                
                # Add display methods as properties
                status_val = test['status']
                test['get_status_display'] = lambda s=status_val: {
                    'requested': 'Requested',
                    'scheduled': 'Scheduled',
                    'in_progress': 'In Progress',
                    'completed': 'Completed',
                    'canceled': 'Canceled'
                }.get(s, s.title() if s else '')
                
                test_type_val = test['test_type']
                test['get_test_type_display'] = lambda t=test_type_val: {
                    'blood': 'Blood Test',
                    'urine': 'Urine Test',
                    'imaging': 'Imaging',
                    'general': 'General',
                    'other': 'Other'
                }.get(t, t.title() if t else '')
                
                # Load related patient info with better error handling
                try:
                    # First check if the auth_user table exists without logging an error
                    check_table_query = """
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='auth_user'
                    """
                    cursor.execute(check_table_query)
                    if not cursor.fetchone():
                        # Table doesn't exist yet, use placeholder data silently
                        test['patient'] = {'user': {'get_full_name': lambda: f"Patient #{test['patient_id']}"}}
                    else:
                        patient_query = """
                        SELECT p.id, u.first_name, u.last_name
                        FROM patient_patient p
                        JOIN auth_user u ON p.user_id = u.id
                        WHERE p.id = ?
                        """
                        cursor.execute(patient_query, [test['patient_id']])
                        patient_row = cursor.fetchone()
                        
                        if patient_row:
                            first_name = patient_row[1]
                            last_name = patient_row[2]
                            test['patient'] = {
                                'id': patient_row[0],
                                'user': {
                                    'get_full_name': lambda fn=first_name, ln=last_name: f"{fn} {ln}"
                                }
                            }
                        else:
                            # Patient not found, use placeholder
                            test['patient'] = {'user': {'get_full_name': lambda: f"Patient #{test['patient_id']}"}}
                except Exception as e:
                    # Only log non-table-missing errors
                    if "auth_user table doesn't exist" not in str(e):
                        print(f"Error loading patient data: {e}")
                    test['patient'] = {'user': {'get_full_name': lambda: f"Patient #{test['patient_id']}"}}
                
                tests.append(test)
            
            return tests
        except Exception as e:
            print(f"Error retrieving lab tests: {e}")
            import traceback
            print(traceback.format_exc())
            return []
