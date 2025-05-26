import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load Knowledge Base
with open("diseases.json", "r") as f:
    diseases_data = json.load(f)
with open("symptoms.json", "r") as f:
    symptoms_data = json.load(f)

diseases = [disease["name"] for disease in diseases_data["diseases"]]
symptom_names = symptoms_data["symptoms"]

# Generate training data
def generate_training_data(diseases_data, symptom_names):
    X_train = []
    y_train = []
    for idx, disease in enumerate(diseases_data["diseases"]):
        symptoms_vector = [1 if symptom in disease["symptoms"] else 0 for symptom in symptom_names]
        X_train.append(symptoms_vector)
        y_train.append(idx)
    return np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.int64)

X_train, y_train = generate_training_data(diseases_data, symptom_names)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

# Define neural network model
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

# Initialize model, loss, and optimizer
model = HealthDiagnosisModel(input_size=len(symptom_names), output_size=len(diseases))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
model.train()
for epoch in range(300):
    optimizer.zero_grad()
    outputs = model(X_train, training=True)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/300], Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "health_diagnosis_model.pth")
print("Model saved to health_diagnosis_model.pth")

# Save symptom names and diseases for inference
with open("model_metadata.json", "w") as f:
    json.dump({"symptom_names": symptom_names, "diseases": diseases}, f)
print("Model metadata saved to model_metadata.json")