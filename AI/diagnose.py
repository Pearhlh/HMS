import json
import numpy as np
import torch
import torch.nn as nn

# Load Knowledge Base and metadata
with open("diseases.json", "r") as f:
    diseases_data = json.load(f)
with open("symptoms.json", "r") as f:
    symptoms_data = json.load(f)
with open("model_metadata.json", "r") as f:
    model_metadata = json.load(f)

symptom_names = model_metadata["symptom_names"]
diseases = model_metadata["diseases"]

# Define simple symptoms to ask
simple_symptoms = [
    "Fever", "Cough", "Sneezing", "Fatigue", "Sore Throat",
    "Headache", "Runny Nose", "Loss of Taste", "Shortness of Breath", "Rash"
]

# Define neural network model (must match training model)
class HealthDiagnosisModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(HealthDiagnosisModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.output = nn.Linear(32, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, training=False):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x) if training else x
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x) if training else x
        x = self.output(x)
        x = self.softmax(x)
        return x

# Load model
model = HealthDiagnosisModel(input_size=len(symptom_names), output_size=len(diseases))
model.load_state_dict(torch.load("health_diagnosis_model.pth"))
model.eval()
print("Model loaded from health_diagnosis_model.pth")

# Predict with uncertainty
def predict_with_uncertainty(model, x, n_iter=100):
    preds = []
    with torch.no_grad():
        for _ in range(n_iter):
            model.train()  # Enable dropout for uncertainty
            pred = model(x, training=True)
            preds.append(pred.numpy())
    preds = np.array(preds)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std

# Main function to run diagnosis
def run_health_assistant():
    print("Chào bạn! Tôi là trợ lý sức khỏe AI.")
    print("Vui lòng trả lời các câu hỏi sau bằng Y/N (Có/Không):")
    
    # Collect user symptoms for simple symptoms
    input_symptoms_simple = []
    for name in simple_symptoms:
        ans = input(f"Bạn có triệu chứng {name}? (Y/N): ").strip().lower()
        input_symptoms_simple.append(1 if ans == 'y' else 0)
    
    # Map simple symptoms to full symptom vector
    input_symptoms_full = [0] * len(symptom_names)
    for i, simple_symptom in enumerate(simple_symptoms):
        if simple_symptom in symptom_names:
            idx = symptom_names.index(simple_symptom)
            input_symptoms_full[idx] = input_symptoms_simple[i]
    
    # Predict disease
    input_array = torch.tensor([input_symptoms_full], dtype=torch.float32)
    mean_probs, std_probs = predict_with_uncertainty(model, input_array)
    most_likely = np.argmax(mean_probs)
    diagnosis = diseases[most_likely]
    diagnosis_info = next(d for d in diseases_data["diseases"] if d["name"] == diagnosis)
    
    # Display results
    print("\nKết quả chẩn đoán với xác suất và độ không chắc chắn:")
    for i, dis in enumerate(diseases):
        print(f"{dis}: Xác suất={mean_probs[0][i]:.3f}, Độ không chắc chắn={std_probs[0][i]:.3f}")
    
    print(f"\nChẩn đoán: {diagnosis} (±{std_probs[0][most_likely]:.3f})")
    print(f"Mức độ nghiêm trọng: {diagnosis_info['severity']}")
    print(f"Xét nghiệm đề xuất: {', '.join(diagnosis_info['tests'])}")
    print(f"Thuốc/Điều trị đề xuất: {', '.join(diagnosis_info['treatments'])}")
    print(f"Lời khuyên: {diagnosis_info['advice']}")
    print("\nLưu ý: Đây chỉ là chẩn đoán sơ bộ. Vui lòng tham khảo bác sĩ để được kiểm tra chính xác.")

# Run the assistant
if __name__ == "__main__":
    run_health_assistant()