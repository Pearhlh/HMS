from rest_framework import viewsets, permissions, status
from rest_framework.response import Response
from rest_framework.decorators import action
from django.shortcuts import get_object_or_404
from django.utils import timezone
from users.models import User
from patient.models import Patient, MedicalRecord
from doctor.models import Doctor, Specialization
from appointment.models import Appointment
from prescription.models import Prescription, PrescriptionItem, Medication
from laboratory.models import LabTest, TestResult
from insurance.models import InsuranceProvider, InsurancePolicy, InsuranceClaim
from billing.models import Bill, Payment
from .serializers import (
    UserSerializer, PatientSerializer, MedicalRecordSerializer, 
    DoctorSerializer, AppointmentSerializer, PrescriptionSerializer,
    PrescriptionItemSerializer, LabTestSerializer, TestResultSerializer,
    InsuranceProviderSerializer, InsurancePolicySerializer, InsuranceClaimSerializer,
    BillSerializer, PaymentSerializer, MedicationSerializer
)
from .schema import HealthcareAPISchema

class IsOwnerOrMedicalStaff(permissions.BasePermission):
    """
    Custom permission to only allow owners of an object or medical staff to access it.
    """
    def has_object_permission(self, request, view, obj):
        user = request.user
        
        # Medical staff can access all records
        if user.is_doctor or user.is_nurse or user.is_admin:
            return True
        
        # Patients can only access their own records
        if hasattr(obj, 'user') and obj.user == user:
            return True
        
        if hasattr(obj, 'patient'):
            if hasattr(obj.patient, 'user') and obj.patient.user == user:
                return True
        
        return False


class UserViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for viewing user information"""
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        # Check if this is a schema generation request
        if getattr(self, 'swagger_fake_view', False):
            return User.objects.none()
            
        user = self.request.user
        if user.is_admin or user.is_superuser:
            return User.objects.all()
        return User.objects.filter(id=user.id)


class PatientViewSet(viewsets.ModelViewSet):
    """
    API endpoints for managing patients.
    
    retrieve:
    Return a specific patient by ID
    
    list:
    Return a list of all patients accessible to the user
    
    create:
    Create a new patient
    
    update:
    Update an existing patient
    
    partial_update:
    Partially update an existing patient
    
    destroy:
    Delete a patient
    """
    serializer_class = PatientSerializer
    permission_classes = [permissions.IsAuthenticated, IsOwnerOrMedicalStaff]
    schema = HealthcareAPISchema()
    
    def get_queryset(self):
        # Check if this is a schema generation request
        if getattr(self, 'swagger_fake_view', False):
            return Patient.objects.none()
            
        user = self.request.user
        if user.is_doctor or user.is_nurse or user.is_admin or user.is_superuser:
            return Patient.objects.all()
        if user.is_patient:
            return Patient.objects.filter(user=user)
        return Patient.objects.none()
    
    def list(self, request, *args, **kwargs):
        """
        List all patients accessible to the current user.
        
        Doctors, nurses, admins, and superusers can see all patients.
        Patients can only see their own records.
        """
        return super().list(request, *args, **kwargs)
    
    def retrieve(self, request, *args, **kwargs):
        """
        Retrieve a specific patient record.
        
        Doctors, nurses, admins, and superusers can access any patient.
        Patients can only access their own record.
        """
        return super().retrieve(request, *args, **kwargs)


class MedicalRecordViewSet(viewsets.ModelViewSet):
    """
    API endpoints for managing medical records.
    
    retrieve:
    Return a specific medical record by ID
    
    list:
    Return a list of all medical records accessible to the user
    
    create:
    Create a new medical record
    
    update:
    Update an existing medical record
    
    partial_update:
    Partially update an existing medical record
    
    destroy:
    Delete a medical record
    """
    serializer_class = MedicalRecordSerializer
    permission_classes = [permissions.IsAuthenticated, IsOwnerOrMedicalStaff]
    schema = HealthcareAPISchema()
    
    def get_queryset(self):
        # Check if this is a schema generation request
        if getattr(self, 'swagger_fake_view', False):
            return MedicalRecord.objects.none()
            
        user = self.request.user
        
        if user.is_doctor or user.is_nurse or user.is_admin or user.is_superuser:
            return MedicalRecord.objects.all()
        
        if user.is_patient:
            try:
                return MedicalRecord.objects.filter(patient__user=user)
            except:
                pass
        
        return MedicalRecord.objects.none()
    
    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)


class DoctorViewSet(viewsets.ModelViewSet):
    """ViewSet for doctors"""
    queryset = Doctor.objects.all()
    serializer_class = DoctorSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_permissions(self):
        if self.action in ['update', 'partial_update', 'destroy']:
            self.permission_classes = [permissions.IsAuthenticated, IsOwnerOrMedicalStaff]
        return super().get_permissions()


class AppointmentViewSet(viewsets.ModelViewSet):
    """ViewSet for appointments"""
    serializer_class = AppointmentSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        # Check if this is a schema generation request
        if getattr(self, 'swagger_fake_view', False):
            return Appointment.objects.none()
            
        user = self.request.user
        
        if user.is_admin or user.is_superuser:
            return Appointment.objects.all()
        
        if user.is_doctor:
            try:
                return Appointment.objects.filter(doctor__user=user)
            except:
                pass
        
        if user.is_patient:
            try:
                return Appointment.objects.filter(patient__user=user)
            except:
                pass
        
        return Appointment.objects.none()
    
    @action(detail=True, methods=['post'])
    def change_status(self, request, pk=None):
        """Change the status of an appointment"""
        appointment = self.get_object()
        new_status = request.data.get('status')
        
        if new_status not in dict(Appointment.STATUS_CHOICES):
            return Response({"error": "Invalid status"}, status=status.HTTP_400_BAD_REQUEST)
        
        appointment.status = new_status
        appointment.save()
        
        serializer = self.get_serializer(appointment)
        return Response(serializer.data)


class PrescriptionViewSet(viewsets.ModelViewSet):
    """ViewSet for prescriptions"""
    serializer_class = PrescriptionSerializer
    permission_classes = [permissions.IsAuthenticated, IsOwnerOrMedicalStaff]
    
    def get_queryset(self):
        # Check if this is a schema generation request
        if getattr(self, 'swagger_fake_view', False):
            return Prescription.objects.none()
            
        user = self.request.user
        
        if user.is_doctor or user.is_pharmacist or user.is_admin or user.is_superuser:
            return Prescription.objects.all()
        
        if user.is_patient:
            try:
                return Prescription.objects.filter(patient__user=user)
            except:
                pass
        
        return Prescription.objects.none()


class LabTestViewSet(viewsets.ModelViewSet):
    """ViewSet for lab tests"""
    serializer_class = LabTestSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        # Check if this is a schema generation request
        if getattr(self, 'swagger_fake_view', False):
            return LabTest.objects.none()
            
        user = self.request.user
        
        if user.is_lab_technician or user.is_doctor or user.is_admin or user.is_superuser:
            return LabTest.objects.all()
        
        if user.is_patient:
            try:
                return LabTest.objects.filter(patient__user=user)
            except:
                pass
        
        return LabTest.objects.none()


class InsuranceClaimViewSet(viewsets.ModelViewSet):
    """ViewSet for insurance claims"""
    serializer_class = InsuranceClaimSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        # Check if this is a schema generation request
        if getattr(self, 'swagger_fake_view', False):
            return InsuranceClaim.objects.none()
            
        user = self.request.user
        
        if user.is_insurance or user.is_admin or user.is_superuser:
            return InsuranceClaim.objects.all()
        
        if user.is_patient:
            try:
                return InsuranceClaim.objects.filter(patient__user=user)
            except:
                pass
        
        return InsuranceClaim.objects.none()
    
    @action(detail=True, methods=['post'])
    def process_claim(self, request, pk=None):
        """Process an insurance claim"""
        claim = self.get_object()
        
        # Only insurance providers can process claims
        if not request.user.is_insurance and not request.user.is_admin and not request.user.is_superuser:
            return Response({"error": "Only insurance providers can process claims"}, 
                           status=status.HTTP_403_FORBIDDEN)
        
        approval_status = request.data.get('approval_status')
        approved_amount = request.data.get('approved_amount')
        notes = request.data.get('notes', '')
        
        if approval_status not in ['approved', 'rejected', 'partial']:
            return Response({"error": "Invalid approval status"}, 
                           status=status.HTTP_400_BAD_REQUEST)
        
        claim.approval_status = approval_status
        
        if approval_status in ['approved', 'partial']:
            try:
                claim.approved_amount = float(approved_amount)
            except (TypeError, ValueError):
                return Response({"error": "Invalid approved amount"}, 
                               status=status.HTTP_400_BAD_REQUEST)
        
        claim.notes = notes
        claim.processed_date = timezone.now()
        claim.processed_by = request.user
        claim.save()
        
        serializer = self.get_serializer(claim)
        return Response(serializer.data)


class BillViewSet(viewsets.ModelViewSet):
    """
    API endpoints for managing bills.
    
    retrieve:
    Return a specific bill by ID
    
    list:
    Return a list of all bills accessible to the user
    
    create:
    Create a new bill
    
    update:
    Update an existing bill
    
    partial_update:
    Partially update an existing bill
    
    destroy:
    Delete a bill
    """
    serializer_class = BillSerializer
    permission_classes = [permissions.IsAuthenticated]
    schema = HealthcareAPISchema()
    
    def get_queryset(self):
        # Check if this is a schema generation request
        if getattr(self, 'swagger_fake_view', False):
            return Bill.objects.none()
            
        user = self.request.user
        if user.is_admin or user.is_staff:
            return Bill.objects.all()
        if user.is_patient:
            return Bill.objects.filter(patient__user=user)
        return Bill.objects.none()


class PaymentViewSet(viewsets.ModelViewSet):
    """
    API endpoints for managing payments.
    
    retrieve:
    Return a specific payment by ID
    
    list:
    Return a list of all payments accessible to the user
    
    create:
    Create a new payment
    
    update:
    Update an existing payment
    
    partial_update:
    Partially update an existing payment
    
    destroy:
    Delete a payment
    """
    serializer_class = PaymentSerializer
    permission_classes = [permissions.IsAuthenticated]
    schema = HealthcareAPISchema()
    
    def get_queryset(self):
        # Check if this is a schema generation request
        if getattr(self, 'swagger_fake_view', False):
            return Payment.objects.none()
            
        user = self.request.user
        if user.is_admin or user.is_staff:
            return Payment.objects.all()
        if user.is_patient:
            return Payment.objects.filter(bill__patient__user=user)
        return Payment.objects.none()
