import pandas as pd
import torch
from torch.utils.data import Dataset
import json
import numpy as np

class MCQADataset(Dataset):
    def __init__(self, file_path, use_context=False, is_json=False):
        """
        Initialize the MCQA dataset.
        """
        self.use_context = use_context
        
        if is_json:
            # Load data from JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    # Try JSONL format
                    data = []
                    f.seek(0)
                    for line in f:
                        if line.strip():
                            try:
                                data.append(json.loads(line.strip()))
                            except json.JSONDecodeError:
                                continue
            
            # Print a sample for debugging
            if data:
                print(f"Sample JSON record: {data[0]}")
            
            # Convert JSON data to a format compatible with the existing code
            self.dataset = []
            for item in data:
                record = {}
                
                # Extract question
                if 'question' in item:
                    record['question'] = item['question']
                else:
                    record['question'] = "No question found"
                
                # Extract options specifically for the MedMCQA format
                options = []
                if 'opa' in item and 'opb' in item and 'opc' in item and 'opd' in item:
                    options = [item['opa'], item['opb'], item['opc'], item['opd']]
                elif 'options' in item and isinstance(item['options'], list):
                    options = item['options']
                else:
                    # Default placeholder if no options found
                    options = ["Option A", "Option B", "Option C", "Option D"]
                
                record['options'] = options
                
                # Extract correct answer
                if 'cop' in item:
                    # MedMCQA format uses 1-based indexing for cop
                    # Convert to 0-based for our model
                    label = int(item['cop']) - 1 if isinstance(item['cop'], (int, str)) else 0
                    if label < 0:
                        label = 0  # Ensure valid index
                elif 'answer' in item:
                    label = item['answer'] if isinstance(item['answer'], int) else 0
                else:
                    label = 0  # Default to first option
                
                record['label'] = label
                
                # Extract context if available and needed
                if use_context and 'context' in item:
                    record['context'] = item['context']
                
                self.dataset.append(record)
            
            # Convert list of dicts to DataFrame
            self.dataset = pd.DataFrame(self.dataset)
            print(f"Loaded {len(self.dataset)} items from {file_path}")
        else:
            # Original code for CSV
            self.dataset = pd.read_csv(file_path)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        item = self.dataset.iloc[idx]
        
        # Extract question, options, etc.
        question = item['question']
        options = item['options']
        label = item['label']
        
        if self.use_context and 'context' in item:
            context = item['context']
            return context, question, options, label
        else:
            return question, options, label