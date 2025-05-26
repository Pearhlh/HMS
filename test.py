import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Knowledge base
knowledge_base = {
    "Flu": {
        "symptoms": ["Fever", "Cough", "Fatigue"],
        "test": "Influenza A/B test",
        "medicine": "Oseltamivir (Tamiflu)"
    },
    "Cold": {
        "symptoms": ["Cough", "Sneezing"],
        "test": "Nasal swab",
        "medicine": "Rest, fluids, antihistamines"
    },
    "COVID-19": {
        "symptoms": ["Fever", "Cough", "Loss of Taste"],
        "test": "PCR test",
        "medicine": "Isolation + Paracetamol"
    },
    "Allergy": {
        "symptoms": ["Sneezing", "Itchy Eyes"],
        "test": "Allergy skin test",
        "medicine": "Loratadine or Cetirizine"
    }
}

# Training data
X_train = np.array([
    [1, 1, 0, 1, 0, 0],  # Flu: Fever, Cough, Fatigue
    [0, 1, 1, 0, 0, 0],  # Cold: Cough, Sneezing
    [1, 1, 0, 0, 1, 0],  # COVID-19: Fever, Cough, Loss of Taste
    [0, 0, 1, 0, 0, 1]   # Allergy: Sneezing, Itchy Eyes
], dtype=np.float32)

y_train = np.array([0, 1, 2, 3], dtype=np.int64)  # Labels for diseases
diseases = ["Flu", "Cold", "COVID-19", "Allergy"]
symptom_names = ["Fever", "Cough", "Sneezing", "Fatigue", "Loss of Taste", "Itchy Eyes"]

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

# Define neural network model
class HealthDiagnosisModel(nn.Module):
    def __init__(self):
        super(HealthDiagnosisModel, self).__init__()
        self.layer1 = nn.Linear(6, 16)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(16, 16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.output = nn.Linear(16, 4)
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

# Initialize model, loss, and optimizer
model = HealthDiagnosisModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train, training=True)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# Predict with uncertainty
def predict_with_uncertainty(model, x, n_iter=100):
    model.eval()
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
    
    # Collect user symptoms
    input_symptoms = []
    for name in symptom_names:
        ans = input(f"Bạn có triệu chứng {name}? (Y/N): ").strip().lower()
        input_symptoms.append(1 if ans == 'y' else 0)
    
    # Predict disease
    input_array = torch.tensor([input_symptoms], dtype=torch.float32)
    mean_probs, std_probs = predict_with_uncertainty(model, input_array)
    most_likely = np.argmax(mean_probs)
    diagnosis = diseases[most_likely]
    
    # Display results
    print("\nKết quả chẩn đoán với xác suất và độ không chắc chắn:")
    for i, dis in enumerate(diseases):
        print(f"{dis}: Xác suất={mean_probs[0][i]:.3f}, Độ không chắc chắn={std_probs[0][i]:.3f}")
    
    print(f"\nChẩn đoán: {diagnosis} (±{std_probs[0][most_likely]:.3f})")
    print(f"Xét nghiệm đề xuất: {knowledge_base[diagnosis]['test']}")
    print(f"Thuốc/Điều trị đề xuất: {knowledge_base[diagnosis]['medicine']}")
    print("\nLưu ý: Đây chỉ là chẩn đoán sơ bộ. Vui lòng tham khảo bác sĩ để được kiểm tra chính xác.")

# Run the assistant
if __name__ == "__main__":
    run_health_assistant()