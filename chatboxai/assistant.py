import os
import torch
from transformers import AutoTokenizer
import logging
import glob
from django.conf import settings
from .model import MCQAModel

# Configure logger
logger = logging.getLogger(__name__)

class HealthcareAssistant:
    """
    AI assistant that uses the trained medical MCQ model to answer healthcare questions
    """
    
    def __init__(self, patient=None, user=None):
        """Initialize with patient context and load the model"""
        self.patient = patient
        self.user = user
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained medical MCQ model"""
        try:
            # Correct path to the model directory
            MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                    'chatboxai', 'models', 
                                    'bert-base-uncased@@@train@@@use_contextFalse@@@seqlen192')
            
            # Find checkpoint file(s) - it has epoch info in the filename
            checkpoint_files = glob.glob(os.path.join(MODEL_DIR, '*.ckpt'))
            
            if not checkpoint_files:
                logger.error(f"No checkpoint files found in {MODEL_DIR}")
                return
                
            # Use the first checkpoint file found
            checkpoint_path = checkpoint_files[0]
            logger.info(f"Using checkpoint: {checkpoint_path}")
            
            # Handle hparams file - it may not exist, so we'll use None if not found
            hparams_path = os.path.join(MODEL_DIR, 'hparams.yaml')
            hparams_arg = hparams_path if os.path.exists(hparams_path) else None
            
            # Load model from checkpoint
            self.model = MCQAModel.load_from_checkpoint(
                checkpoint_path,
                hparams_file=hparams_arg
            )
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            logger.info("Medical MCQ model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
    
    def get_response(self, query, chat_history=None):
        """Generate a response based on the trained medical MCQ model"""
        if not self.model or not self.tokenizer:
            return "I'm currently unable to answer medical questions. Please try again later."
        
        try:
            # Generate 4 standard options if not provided
            # This converts any query into an MCQ format our model can process
            options = [
                "This condition is normal and no treatment is needed",
                "This requires medication and should be treated by a doctor",
                "This is a medical emergency requiring immediate attention",
                "More information is needed to provide proper guidance"
            ]
            
            # Process through our trained model
            predictions = self.predict(query, options)
            
            # Return the best answer from our model
            return predictions["answer"]
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return "I apologize, but I couldn't process your medical question properly. Please try rephrasing your question."
    
    def predict(self, question, options):
        """Use the trained model to predict the best answer from options"""
        # Format input as our model expects (question + option pairs)
        question_option_pairs = [f"{question} {option}" for option in options]
        
        # Tokenize inputs for the model
        batch_encoding = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": []
        }
        
        for q_opt in question_option_pairs:
            encoded = self.tokenizer.encode_plus(
                q_opt,
                max_length=192,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            for key in batch_encoding:
                batch_encoding[key].append(encoded[key].squeeze(0))
        
        # Convert lists to tensors
        for key in batch_encoding:
            batch_encoding[key] = torch.stack(batch_encoding[key])
        
        # Get model prediction
        with torch.no_grad():
            logits = self.model(**batch_encoding)
            probs = torch.softmax(logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
        
        # Convert to standard Python types
        prediction_idx = prediction.item()
        confidence_score = confidence.item()
        
        # Format the response
        answer = options[prediction_idx]
        result = {
            "prediction": prediction_idx,
            "confidence": confidence_score,
            "answer": f"{answer} (Confidence: {confidence_score:.1%})"
        }
        
        return result
