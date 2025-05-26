from rest_framework import serializers
from users.models import User
from patient.models import Patient, MedicalRecord
from doctor.models import Doctor, Specialization
from appointment.models import Appointment
from prescription.models import Prescription, PrescriptionItem, Medication  # Added Medication import
from laboratory.models import LabTest, TestResult
from insurance.models import InsuranceProvider, InsurancePolicy, InsuranceClaim
from billing.models import Bill, Payment  # Changed Invoice to Bill

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'email', 'first_name', 'last_name', 'role', 'date_joined']
        read_only_fields = ['date_joined']


class PatientSerializer(serializers.ModelSerializer):
    user_details = UserSerializer(source='user', read_only=True)
    
    class Meta:
        model = Patient
        fields = [
            'id', 'user', 'user_details', 
            'emergency_contact_name', 'emergency_contact_phone',
            'blood_type', 'gender', 
            'allergies', 
        ]


class MedicalRecordSerializer(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='patient.user.get_full_name', read_only=True)
    doctor_name = serializers.CharField(source='created_by.get_full_name', read_only=True)
    
    class Meta:
        model = MedicalRecord
        fields = [
            'id', 'patient', 'patient_name', 'record_date', 
            'diagnosis', 'treatment', 'notes',
            'created_by', 'doctor_name'
        ]
        read_only_fields = ['record_date', 'created_by']


class SpecializationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Specialization
        fields = ['id', 'name', 'description']


class DoctorSerializer(serializers.ModelSerializer):
    user_details = UserSerializer(source='user', read_only=True)
    specialization_name = serializers.CharField(source='specialization.name', read_only=True)
    
    class Meta:
        model = Doctor
        fields = [
            'id', 'user', 'user_details', 'license_number', 'specialization',
            'specialization_name', 'qualifications', 'experience_years',
            'consultation_fee', 'available_days', 'available_hours_start',
            'available_hours_end'
        ]


class AppointmentSerializer(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='patient.user.get_full_name', read_only=True)
    doctor_name = serializers.CharField(source='doctor.user.get_full_name', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    
    class Meta:
        model = Appointment
        fields = [
            'id', 'patient', 'patient_name', 'doctor', 'doctor_name',
            'appointment_date', 'appointment_time', 'status', 'status_display',
            'reason', 'notes', 'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']


class MedicationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Medication
        fields = ['id', 'name', 'description', 'dosage_form', 'strength', 'manufacturer']


class PrescriptionItemSerializer(serializers.ModelSerializer):
    medication_details = MedicationSerializer(source='medication', read_only=True)
    
    class Meta:
        model = PrescriptionItem
        fields = [
            'id', 'prescription', 'medication', 'medication_details',
            'dosage', 'frequency', 'duration', 'instructions'
        ]


class PrescriptionSerializer(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='patient.user.get_full_name', read_only=True)
    doctor_name = serializers.CharField(source='doctor.user.get_full_name', read_only=True)
    items = PrescriptionItemSerializer(many=True, read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    
    class Meta:
        model = Prescription
        fields = [
            'id', 'patient', 'patient_name', 'doctor', 'doctor_name',
            'date_prescribed', 'diagnosis', 'notes', 'status',
            'status_display', 'items'
        ]
        read_only_fields = ['date_prescribed']


class TestResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = TestResult
        fields = [
            'id', 'lab_test', 'result_value', 'reference_range',
            'is_abnormal', 'notes', 'uploaded_at'
        ]
        read_only_fields = ['uploaded_at']


class LabTestSerializer(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='patient.user.get_full_name', read_only=True)
    doctor_name = serializers.CharField(source='requested_by.user.get_full_name', read_only=True)  # Changed source from doctor to requested_by
    test_results = TestResultSerializer(many=True, read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    
    class Meta:
        model = LabTest
        fields = [
            'id', 'patient', 'patient_name', 'requested_by', 'doctor_name',  # Changed doctor to requested_by
            'test_name', 'test_date', 'status', 'status_display', 
            'notes', 'results', 'test_results'  # Changed report_file to results to match model
        ]


class InsuranceProviderSerializer(serializers.ModelSerializer):
    class Meta:
        model = InsuranceProvider
        fields = [
            'id', 'name', 'contact_person', 'email', 'phone_number', 
            'address', 'website'
        ]


class InsurancePolicySerializer(serializers.ModelSerializer):
    provider_name = serializers.CharField(source='provider.name', read_only=True)
    patient_name = serializers.CharField(source='patient.user.get_full_name', read_only=True)
    
    class Meta:
        model = InsurancePolicy
        fields = [
            'id', 'policy_number', 'provider', 'provider_name', 
            'patient', 'patient_name', 'start_date', 'end_date',
            'coverage_amount', 'coverage_percentage', 'status'
        ]


class InsuranceClaimSerializer(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='patient.user.get_full_name', read_only=True)
    policy_number = serializers.CharField(source='insurance_policy.policy_number', read_only=True)
    provider_name = serializers.CharField(source='insurance_policy.provider.name', read_only=True)
    
    class Meta:
        model = InsuranceClaim
        fields = [
            'id', 'claim_number', 'patient', 'patient_name', 
            'insurance_policy', 'policy_number', 'provider_name',
            'service_date', 'claim_date', 'diagnosis_codes',
            'service_codes', 'claim_amount', 'approved_amount',
            'approval_status', 'notes', 'processed_date', 'processed_by'
        ]
        read_only_fields = ['claim_number', 'processed_date', 'processed_by']


class BillSerializer(serializers.ModelSerializer):
    class Meta:
        model = Bill
        fields = ['id', 'invoice_number', 'patient', 'amount', 'due_date', 'status', 'created_date']


class PaymentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Payment
        fields = ['id', 'bill', 'amount', 'payment_date', 'payment_method', 'status']
