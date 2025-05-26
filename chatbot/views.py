from django.shortcuts import render
from .forms import SymptomForm
from .diagnosis import HealthDiagnosisBot
from django.contrib.auth.decorators import login_required

@login_required
def symptom_form(request):
    form = SymptomForm()
    return render(request, 'symptom_form.html', {'form': form})
@login_required
def diagnose(request):
    if request.method == 'POST':
        form = SymptomForm(request.POST)
        if form.is_valid():
            symptoms = form.cleaned_data
            bot = HealthDiagnosisBot()
            diagnosis_result = bot.diagnose(symptoms)
            return render(request, 'diagnosis_result.html', {'result': diagnosis_result})
    return render(request, 'symptom_form.html', {'form': SymptomForm()})