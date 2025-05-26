import pytorch_lightning as pl

from pytorch_lightning import Trainer
from torch.utils.data import SequentialSampler,RandomSampler
from torch import nn
import numpy as np
import math
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader,RandomSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer,AutoModel
import functools

class MCQAModel(pl.LightningModule):
  def __init__(self,
               model_name_or_path,
               args):
    
    super().__init__()
    self.init_encoder_model(model_name_or_path)
    self.args = args
    self.batch_size = self.args['batch_size']
    self.dropout = nn.Dropout(self.args['hidden_dropout_prob'])
    self.linear = nn.Linear(in_features=self.args['hidden_size'],out_features=1)
    self.ce_loss = nn.CrossEntropyLoss()
    self.save_hyperparameters()
    # Initialize attributes to store validation outputs
    self.validation_step_outputs = []
    self.test_step_outputs = []
    
  def init_encoder_model(self,model_name_or_path):
    self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    self.model = AutoModel.from_pretrained(model_name_or_path)
 
  def prepare_dataset(self,train_dataset,val_dataset,test_dataset=None):
    """
    helper to set the train and val dataset. Doing it during class initialization
    causes issues while loading checkpoint as the dataset class needs to be 
    present for the weights to be loaded.
    """
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset
    if test_dataset != None:
        self.test_dataset = test_dataset
    else:
        self.test_dataset = val_dataset
  
  def forward(self, input_ids, attention_mask, token_type_ids):
    outputs = self.model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)
    
    pooled_output = outputs[1]
    pooled_output = self.dropout(pooled_output)
    logits = self.linear(pooled_output)
    
    # Get the number of options from the args or default to 4
    num_choices = self.args.get('num_choices', 4)
    
    # Ensure that the batch size is divisible by num_choices
    batch_size = logits.size(0) // num_choices
    
    # Safety check to avoid shape errors
    if batch_size * num_choices != logits.size(0):
        print(f"WARNING: Input size {logits.size(0)} is not divisible by num_choices {num_choices}")
        # Don't reshape if sizes don't align
        return logits
        
    # Reshape logits to [batch_size, num_choices]
    reshaped_logits = logits.view(batch_size, num_choices)
    print(f"DEBUG: Reshaped logits: {reshaped_logits.shape}")
    return reshaped_logits
  
  def training_step(self, batch, batch_idx):
    inputs, labels = batch
    for key in inputs:
      # Safely move tensors to the appropriate device
      if hasattr(self, 'device') and self.device:
        device = self.device
      elif 'device' in self.args and self.args['device']:
        device = self.args['device']
      else:
        device = 'cpu'
        
      # Check if device is valid (avoid CUDA if not available)
      if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
      
      inputs[key] = inputs[key].to(device)
    logits = self(**inputs)
    loss = self.ce_loss(logits, labels)
    self.log('train_loss', loss, on_epoch=True)
    return loss
  
  def test_step(self, batch, batch_idx):
    inputs, labels = batch
    for key in inputs:
      # Use safe device handling
      device = 'cpu'
      if 'device' in self.args and self.args['device'] and \
         (self.args['device'] != 'cuda' or torch.cuda.is_available()):
        device = self.args['device']
      inputs[key] = inputs[key].to(device)
    logits = self(**inputs)
    loss = self.ce_loss(logits,labels)
    self.log('test_loss', loss, on_epoch=True)
    # Store outputs in a list that can be accessed in on_test_epoch_end
    output = {'test_loss': loss, 'logits': logits, 'labels': labels}
    self.test_step_outputs.append(output)
    return output
 
  def on_test_epoch_end(self):
    # Process the collected outputs
    avg_loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean()
    all_logits = torch.cat([x['logits'] for x in self.test_step_outputs])
    all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
    
    predictions = torch.argmax(all_logits, axis=-1)
    self.test_predictions = predictions
    correct_predictions = torch.sum(predictions == all_labels)
    accuracy = correct_predictions.cpu().detach().numpy() / predictions.size()[0]
    
    self.log('avg_test_loss', avg_loss)
    self.log('test_acc', accuracy)
    
    # Clear the outputs list for the next epoch
    self.test_step_outputs.clear()
    
    return {'test_loss': avg_loss, 'test_acc': accuracy}
  
  def validation_step(self, batch, batch_idx):
    inputs, labels = batch
    for key in inputs:
      # Use safe device handling
      device = 'cpu'
      if 'device' in self.args and self.args['device'] and \
         (self.args['device'] != 'cuda' or torch.cuda.is_available()):
        device = self.args['device']
      inputs[key] = inputs[key].to(device)
    logits = self(**inputs)
    loss = self.ce_loss(logits,labels)
    self.log('val_loss', loss, on_epoch=True)
    
    # Store outputs in a list that can be accessed in on_validation_epoch_end
    output = {'val_loss': loss, 'logits': logits, 'labels': labels}
    self.validation_step_outputs.append(output)
    return output

  # Replace validation_epoch_end with on_validation_epoch_end
  def on_validation_epoch_end(self):
    # Process the collected outputs
    avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
    all_logits = torch.cat([x['logits'] for x in self.validation_step_outputs])
    all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
    
    predictions = torch.argmax(all_logits, axis=-1)
    correct_predictions = torch.sum(predictions == all_labels)
    accuracy = correct_predictions.cpu().detach().numpy() / predictions.size()[0]
    
    self.log('avg_val_loss', avg_loss)
    self.log('val_acc', accuracy)
    
    # Clear the outputs list for the next epoch
    self.validation_step_outputs.clear()
    
    return {'val_loss': avg_loss, 'val_acc': accuracy}
        
  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(),lr=self.args['learning_rate'],eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=(self.args['num_epochs'] + 1) * math.ceil(len(self.train_dataset) / self.args['batch_size']),
    )
    return [optimizer],[scheduler]
  
  def process_batch(self, batch, tokenizer, max_len=32):
    expanded_batch = []
    labels = []
    context = None
    
    # Debug batch info
    if len(batch) > 0:
      print(f"DEBUG: Batch size: {len(batch)}")
      print(f"DEBUG: First item type: {type(batch[0])}")
      print(f"DEBUG: First item structure: {batch[0]}")
    
    # Number of options per question (default to 4 for multiple choice)
    num_options = self.args.get('num_choices', 4)
    
    for data_tuple in batch:
        # Extract data
        if len(data_tuple) == 4:
          context, question, options, label = data_tuple
        else:
          question, options, label = data_tuple
          
        # Ensure label is an integer
        if not isinstance(label, (int, np.integer)):
          try:
            label = int(label)
          except:
            label = 0
        
        # Ensure options is a list with exactly num_options items
        if not isinstance(options, list):
          options = [str(options)] * num_options
        elif len(options) < num_options:
          # Pad with empty strings if fewer than num_options
          options = options + [""] * (num_options - len(options))
        elif len(options) > num_options:
          # Trim if more than num_options
          options = options[:num_options]
        
        # Create question-option pairs
        question_option_pairs = [f"{question} {option}" for option in options]
        
        # Add one label per question (not per option)
        labels.append(label)
        
        # Add question-option pairs to batch
        if context:
          for q_opt in question_option_pairs:
            expanded_batch.append((context, q_opt))
        else:
          expanded_batch.extend(question_option_pairs)
    
    # Now tokenize the expanded batch
    tokenized_inputs = []
    for item in expanded_batch:
      # Handle context-based vs regular QA
      if isinstance(item, tuple) and len(item) == 2:
        # Context-based: (context, question+option)
        ctx, q_opt = item
        encoded = tokenizer.encode_plus(
          ctx, q_opt,
          truncation=True,
          padding="max_length",
          max_length=max_len,
          return_tensors="pt"
        )
      else:
        # Regular QA: question+option
        encoded = tokenizer.encode_plus(
          item,
          truncation=True,
          padding="max_length",
          max_length=max_len,
          return_tensors="pt"
        )
      
      # Add to list of tokenized inputs
      tokenized_inputs.append({
        key: value.squeeze(0) for key, value in encoded.items()
      })
    
    # Stack individual encodings to create batch tensors
    if tokenized_inputs:
      batch_encoding = {
        key: torch.stack([inp[key] for inp in tokenized_inputs]) 
        for key in tokenized_inputs[0].keys()
      }
    else:
      # Handle empty batch case
      batch_encoding = {
        "input_ids": torch.zeros((0, max_len), dtype=torch.long),
        "attention_mask": torch.zeros((0, max_len), dtype=torch.long),
        "token_type_ids": torch.zeros((0, max_len), dtype=torch.long)
      }
    
    # Create labels tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Debug output info
    print(f"DEBUG: Input size: {next(iter(batch_encoding.values())).shape}, Labels size: {labels_tensor.shape}")
    
    return batch_encoding, labels_tensor
  
  def train_dataloader(self):
    train_sampler = RandomSampler(self.train_dataset)
    model_collate_fn = functools.partial(
      self.process_batch,
      tokenizer=self.tokenizer,
      max_len=self.args['max_len']
      )
    train_dataloader = DataLoader(self.train_dataset,
                                batch_size=self.batch_size,
                                sampler=train_sampler,
                                collate_fn=model_collate_fn)
    return train_dataloader
  
  def val_dataloader(self):
    eval_sampler = SequentialSampler(self.val_dataset)
    model_collate_fn = functools.partial(
      self.process_batch,
      tokenizer=self.tokenizer,
      max_len=self.args['max_len']
      )
    val_dataloader = DataLoader(self.val_dataset,
                                batch_size=self.batch_size,
                                sampler=eval_sampler,
                                collate_fn=model_collate_fn)
    return val_dataloader
  
  def test_dataloader(self):
    eval_sampler = SequentialSampler(self.test_dataset)
    model_collate_fn = functools.partial(
      self.process_batch,
      tokenizer=self.tokenizer,
      max_len=self.args['max_len']
      )
    test_dataloader = DataLoader(self.test_dataset,
                                batch_size=self.batch_size,
                                sampler=eval_sampler,
                                collate_fn=model_collate_fn)
    return test_dataloader
