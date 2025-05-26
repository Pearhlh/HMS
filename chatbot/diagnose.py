import json
import numpy as np
import torch
import torch.nn as nn
import os
import logging

logger = logging.getLogger(__name__)

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

class HealthDiagnosisBot:
    def __init__(self):
        try:
            # Load Knowledge Base and metadata
            with open('chatbot/models/diseases.json', 'r') as f:
                self.diseases_data = json.load(f)
            with open('chatbot/models/symptoms.json', 'r') as f:
                self.symptoms_data = json.load(f)
            with open('chatbot/models/model_metadata.json', 'r') as f:
                self.model_metadata = json.load(f)
        except FileNotFoundError as e:
            logger.error(f"Missing file: {str(e)}")
            raise FileNotFoundError(f"Could not load required files: {str(e)}")

        self.symptom_names = self.model_metadata["symptom_names"]
        self.diseases = self.model_metadata["diseases"]
        self.simple_symptoms = [
            "Fever", "Cough", "Sneezing", "Fatigue", "Sore Throat",
            "Headache", "Runny Nose", "Loss of Taste", "Shortness of Breath", "Rash"
        ]

        # Initialize and load model
        self.model = HealthDiagnosisModel(
            input_size=len(self.symptom_names),
            output_size=len(self.diseases)
        )
        path = 'chatbot/models/health_diagnosis_model.pth'
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        # print(f"Model loaded from {"chatbot/models/health_diagnosis_model.pth"}")

    def predict_with_uncertainty(self, x, n_iter=100):
        """Dự đoán với độ không chắc chắn bằng cách sử dụng dropout."""
        preds = []
        with torch.no_grad():
            for _ in range(n_iter):
                self.model.train()  # Enable dropout for uncertainty
                pred = self.model(x, training=True)
                preds.append(pred.numpy())
        preds = np.array(preds)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return mean, std

    def collect_symptoms(self):
        """Thu thập triệu chứng từ người dùng."""
        print("Chào bạn! Tôi là trợ lý sức khỏe AI.")
        print("Vui lòng trả lời các câu hỏi sau bằng Y/N (Có/Không):")
        
        input_symptoms_simple = []
        for name in self.simple_symptoms:
            ans = input(f"Bạn có triệu chứng {name}? (Y/N): ").strip().lower()
            input_symptoms_simple.append(1 if ans == 'y' else 0)
        
        # Map simple symptoms to full symptom vector
        input_symptoms_full = [0] * len(self.symptom_names)
        for i, simple_symptom in enumerate(self.simple_symptoms):
            if simple_symptom in self.symptom_names:
                idx = self.symptom_names.index(simple_symptom)
                input_symptoms_full[idx] = input_symptoms_simple[i]
        
        return input_symptoms_full

    def diagnose(self, input_symptoms):
        """Chạy chẩn đoán dựa trên triệu chứng đầu vào."""
        input_array = torch.tensor([input_symptoms], dtype=torch.float32)
        mean_probs, std_probs = self.predict_with_uncertainty(input_array)
        most_likely = np.argmax(mean_probs)
        diagnosis = self.diseases[most_likely]
        diagnosis_info = next(d for d in self.diseases_data["diseases"] if d["name"] == diagnosis)
        
        return mean_probs, std_probs, diagnosis, diagnosis_info

    def display_results(self, mean_probs, std_probs, diagnosis, diagnosis_info):
        """Hiển thị kết quả chẩn đoán."""
        
        print(f"\nChẩn đoán: {diagnosis}")
        print(f"Mức độ nghiêm trọng: {diagnosis_info['severity']}")
        print(f"Xét nghiệm đề xuất: {', '.join(diagnosis_info['tests'])}")
        print(f"Thuốc/Điều trị đề xuất: {', '.join(diagnosis_info['treatments'])}")
        print(f"Lời khuyên: {diagnosis_info['advice']}")
        print("\nLưu ý: Đây chỉ là chẩn đoán sơ bộ. Vui lòng tham khảo bác sĩ để được kiểm tra chính xác.")

    def run_diagnosis(self):
        """Chạy toàn bộ quy trình chẩn đoán từ thu thập triệu chứng đến hiển thị kết quả."""
        symptoms = self.collect_symptoms()
        mean_probs, std_probs, diagnosis, diagnosis_info = self.diagnose(symptoms)
        self.display_results(mean_probs, std_probs, diagnosis, diagnosis_info)