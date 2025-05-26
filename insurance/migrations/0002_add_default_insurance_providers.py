from django.db import migrations

def add_default_insurance_providers(apps, schema_editor):
    """
    Add default insurance providers to the database
    """
    InsuranceProvider = apps.get_model('insurance', 'InsuranceProvider')
    
    # Create a list of default providers
    default_providers = [
        {
            'name': 'Blue Cross Blue Shield',
            'address': '225 North Michigan Avenue, Chicago, IL 60601',
            'phone': '1-800-262-2583',
            'email': 'customerservice@bcbs.com',
            'website': 'https://www.bcbs.com',
            'description': 'Blue Cross Blue Shield is a federation of 35 separate United States health insurance companies that provide health insurance in the United States.',
        },
        {
            'name': 'UnitedHealthcare',
            'address': '9900 Bren Road East, Minnetonka, MN 55343',
            'phone': '1-866-633-2446',
            'email': 'customerservice@uhc.com',
            'website': 'https://www.uhc.com',
            'description': 'UnitedHealthcare offers health benefit programs for individuals, employers, and Medicare and Medicaid beneficiaries.',
        },
        {
            'name': 'Aetna',
            'address': '151 Farmington Avenue, Hartford, CT 06156',
            'phone': '1-800-872-3862',
            'email': 'memberservices@aetna.com',
            'website': 'https://www.aetna.com',
            'description': 'Aetna offers healthcare, dental, pharmacy, group life, and disability insurance, and employee benefits.',
        },
        {
            'name': 'Cigna',
            'address': '900 Cottage Grove Road, Bloomfield, CT 06002',
            'phone': '1-800-997-1654',
            'email': 'customerservice@cigna.com',
            'website': 'https://www.cigna.com',
            'description': 'Cigna is a global health service company that offers health, pharmacy, dental, supplemental insurance and Medicare plans.',
        },
        {
            'name': 'Humana',
            'address': '500 West Main Street, Louisville, KY 40202',
            'phone': '1-800-457-4708',
            'email': 'customerservice@humana.com',
            'website': 'https://www.humana.com',
            'description': 'Humana is a health insurance company that markets Medicare Advantage health plans, standalone prescription drug plans, and private health insurance.',
        },
        {
            'name': 'Kaiser Permanente',
            'address': 'One Kaiser Plaza, Oakland, CA 94612',
            'phone': '1-800-464-4000',
            'email': 'info@kp.org',
            'website': 'https://healthy.kaiserpermanente.org',
            'description': 'Kaiser Permanente is an integrated managed care consortium with both an insurance component and healthcare provider component.',
        }
    ]
    
    # Add each provider to the database
    for provider_data in default_providers:
        # Check if provider already exists to avoid duplicates
        if not InsuranceProvider.objects.filter(name=provider_data['name']).exists():
            InsuranceProvider.objects.create(**provider_data)


def remove_default_insurance_providers(apps, schema_editor):
    """
    Remove default insurance providers from the database (for migration rollback)
    """
    InsuranceProvider = apps.get_model('insurance', 'InsuranceProvider')
    
    # List of provider names to remove
    provider_names = [
        'Blue Cross Blue Shield',
        'UnitedHealthcare',
        'Aetna',
        'Cigna',
        'Humana',
        'Kaiser Permanente',
    ]
    
    # Delete each provider
    InsuranceProvider.objects.filter(name__in=provider_names).delete()


class Migration(migrations.Migration):

    dependencies = [
        ('insurance', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(add_default_insurance_providers, remove_default_insurance_providers),
    ]
