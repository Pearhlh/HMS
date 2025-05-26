from conf.args import Arguments
from model import MCQAModel
from dataset import MCQADataset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch, os, sys
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process
import time, argparse
import json
import glob


EXPERIMENT_DATASET_FOLDER = "/content/medmcqa_data/"
WB_PROJECT = "MEDMCQA"

def train(gpu,
          args,
          exp_dataset_folder,
          experiment_name,
          models_folder,
          version,
          use_wandb=True):

    pl.seed_everything(42)
    
    # Enhanced GPU detection and configuration
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_count = torch.cuda.device_count()
        print(f"CUDA is available: {gpu_count} GPU(s) detected")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**2)
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**2)
            print(f"GPU {i}: {gpu_name} | Memory allocated: {memory_allocated:.2f}MB | Reserved: {memory_reserved:.2f}MB")
    else:
        print("CUDA is not available. Training will use CPU only.")
        print("Check PyTorch installation - make sure it includes CUDA support.")
    
    # Performance optimization configurations
    torch.backends.cudnn.benchmark = True  # Speed up training when input sizes don't change
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        # Enable TensorFloat-32 for faster matrix multiplications on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TensorFloat-32 enabled for faster training on compatible GPUs")
    
    # Force GPU usage if specified and available
    if gpu is None and gpu_available:
        gpu = 0  # Default to first GPU if none specified but GPU is available
        print(f"No GPU specified but CUDA available. Using GPU 0.")
    elif gpu and not gpu_available:
        print("Warning: GPU was requested but CUDA is not available. Falling back to CPU.")
        gpu = None
        
    # Set the device in args to ensure consistency
    if gpu_available and gpu is not None:
        args.__dict__['device'] = 'cuda'
        # Set environment variable to prefer GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu if not isinstance(gpu, list) else ','.join(map(str, gpu)))
        print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        args.__dict__['device'] = 'cpu'
    
    required_files = [args.train_csv, args.test_csv, args.dev_csv]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required data file not found: {file_path}")
    
    EXPERIMENT_FOLDER = os.path.join(models_folder, experiment_name)
    os.makedirs(EXPERIMENT_FOLDER, exist_ok=True)
    experiment_string = experiment_name + '-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}'

    dataloader_kwargs = {
        'num_workers': min(os.cpu_count() or 4, 8),
        'pin_memory': True if gpu_available else False,
        'persistent_workers': True if (os.cpu_count() or 4) > 0 else False,
    }
    print(f"DataLoader optimizations: {dataloader_kwargs}")
    args.__dict__['dataloader_kwargs'] = dataloader_kwargs
    
    wb = WandbLogger(project=WB_PROJECT, name=experiment_name, version=version)
    csv_log = CSVLogger(models_folder, name=experiment_name, version=version)

    train_dataset = MCQADataset(args.train_csv, args.use_context, 
                               is_json=args.train_csv.lower().endswith('.json'))
    test_dataset = MCQADataset(args.test_csv, args.use_context, 
                              is_json=args.test_csv.lower().endswith('.json'))
    val_dataset = MCQADataset(args.dev_csv, args.use_context, 
                             is_json=args.dev_csv.lower().endswith('.json'))

    es_callback = pl.callbacks.EarlyStopping(monitor='val_loss',
                                    min_delta=0.00,
                                    patience=2,
                                    verbose=True,
                                    mode='min')
    cp_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                              dirpath=EXPERIMENT_FOLDER,
                                              filename=experiment_string,
                                              save_top_k=1,
                                              save_weights_only=True,
                                              mode='min')
    # Add learning rate monitor
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    mcqaModel = MCQAModel(model_name_or_path=args.pretrained_model_name,
                          args=args.__dict__)
    mcqaModel.prepare_dataset(train_dataset=train_dataset,
                              test_dataset=test_dataset,
                              val_dataset=val_dataset)
    if gpu_available and gpu is not None:
        torch.cuda.set_device(0 if not isinstance(gpu, list) else gpu[0])
        print(f"Active GPU set to: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
        torch.backends.cudnn.benchmark = True
        print("CUDNN benchmark enabled for improved performance")
    
    effective_batch_size = getattr(args, 'batch_size', 16)
    grad_accum_steps = max(1, min(8, effective_batch_size // 16))
    actual_batch_size = effective_batch_size // grad_accum_steps
    print(f"Training with batch size: {actual_batch_size}, gradient accumulation: {grad_accum_steps}")
    
    trainer = Trainer(
        accelerator="gpu" if gpu_available and gpu is not None else "cpu",
        devices=gpu if isinstance(gpu, list) and gpu_available else 1 if gpu_available and gpu is not None else 1,
        strategy="ddp" if isinstance(gpu, list) and len(gpu) > 1 and gpu_available else "auto",
        logger=[wb, csv_log],
        callbacks=[es_callback, cp_callback, lr_monitor],
        max_epochs=args.num_epochs,
        precision="16-mixed" if gpu_available else 32,
        gradient_clip_val=1.0,
        accumulate_grad_batches=grad_accum_steps,
        log_every_n_steps=10,
        deterministic=False,
        enable_progress_bar=True,
        enable_model_summary=True,
        profiler="simple",
    )
    print(f"Training starting with {'16-bit mixed precision' if gpu_available else '32-bit'} and {trainer.accelerator}")
    trainer.fit(mcqaModel)
    print(f"Training completed")

    ckpt = [f for f in os.listdir(EXPERIMENT_FOLDER) if f.endswith('.ckpt')]

    inference_model = MCQAModel.load_from_checkpoint(os.path.join(EXPERIMENT_FOLDER, ckpt[0]))
    if torch.cuda.is_available():
        gpu_id = 0 if not isinstance(gpu, list) else gpu[0]
        inference_model = inference_model.to(f"cuda:{gpu_id}")
        print(f"Inference model moved to GPU {gpu_id}")
    else:
        inference_model = inference_model.to("cpu")
        print("Inference model using CPU")
    inference_model = inference_model.eval()

    with open(args.test_csv, 'r', encoding='utf-8') as f:
        try:
            test_data = json.load(f)
        except json.JSONDecodeError:
            test_data = []
            with open(args.test_csv, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        test_data.append(json.loads(line.strip()))
    
    test_df = pd.DataFrame(test_data)
    test_df.loc[:, "predictions"] = [pred+1 for pred in run_inference(inference_model, mcqaModel.test_dataloader(), args)]
    test_df.to_json(os.path.join(EXPERIMENT_FOLDER, "test_results.json"), orient="records", lines=True)
    print(f"Test predictions written to {os.path.join(EXPERIMENT_FOLDER, 'test_results.json')}")

    with open(args.dev_csv, 'r', encoding='utf-8') as f:
        try:
            val_data = json.load(f)
        except json.JSONDecodeError:
            val_data = []
            with open(args.dev_csv, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        val_data.append(json.loads(line.strip()))
    
    val_df = pd.DataFrame(val_data)
    val_df.loc[:, "predictions"] = [pred+1 for pred in run_inference(inference_model, mcqaModel.val_dataloader(), args)]
    val_df.to_json(os.path.join(EXPERIMENT_FOLDER, "dev_results.json"), orient="records", lines=True)
    print(f"Val predictions written to {os.path.join(EXPERIMENT_FOLDER, 'dev_results.json')}")
    del mcqaModel
    del inference_model
    del trainer
    torch.cuda.empty_cache()
    
def run_inference(model, dataloader, args):
    predictions = []
    for idx, (inputs, labels) in tqdm(enumerate(dataloader)):
        batch_size = len(labels)
        for key in inputs.keys():
            inputs[key] = inputs[key].to(args.device)
        with torch.no_grad():
            outputs = model(**inputs)
        prediction_idxs = torch.argmax(outputs, axis=1).cpu().detach().numpy()
        predictions.extend(list(prediction_idxs))
    return predictions

if __name__ == "__main__":
    os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", "offline")
    models = ["allenai/scibert_scivocab_uncased", "bert-base-uncased"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-uncased", help="name of the model")
    parser.add_argument("--dataset_folder_name", default="/content/medmcqa_data/", help="dataset folder")
    parser.add_argument("--use_context", default=False, action='store_true', help="mention this flag to use_context")
    parser.add_argument("--no_wandb", default=False, action='store_true', help="disable wandb logging")
    parser.add_argument("--force_gpu", default=False, action='store_true', help="force GPU usage even if CUDA reports as unavailable")
    cmd_args = parser.parse_args()
    exp_dataset_folder = cmd_args.dataset_folder_name
    if not os.path.isabs(exp_dataset_folder):
        exp_dataset_folder = os.path.join(EXPERIMENT_DATASET_FOLDER, exp_dataset_folder)
    
    if not os.path.exists(exp_dataset_folder):
        print(f"Error: Dataset folder '{exp_dataset_folder}' does not exist.")
        print("Please make sure the folder exists and contains JSON files for training data.")
        sys.exit(1)
    model = cmd_args.model
    print(f"Training started for model - {model} variant - {exp_dataset_folder} use_context - {str(cmd_args.use_context)}")

    json_files = glob.glob(os.path.join(exp_dataset_folder, "*.json"))
    if not json_files:
        print(f"Error: No JSON files found in '{exp_dataset_folder}'.")
        print("Please make sure the folder contains the required JSON files.")
        sys.exit(1)
    json_file = json_files[0]
    args = Arguments(train_csv=json_file,
                     test_csv=json_file,
                     dev_csv=json_file,
                     pretrained_model_name=model,
                     use_context=cmd_args.use_context)

    if cmd_args.force_gpu:
        print("Force GPU mode enabled - will attempt to use GPU even if automatic detection fails")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        if args.gpu is None:
            args.gpu = 1

    exp_name = f"{model}@@@{os.path.basename(exp_dataset_folder)}@@@use_context{str(cmd_args.use_context)}@@@seqlen{str(args.max_len)}".replace("/", "_")

    train(gpu=args.gpu,
          args=args,
          exp_dataset_folder=exp_dataset_folder,
          experiment_name=exp_name,
          models_folder="./models",
          version=exp_name,
          use_wandb=not cmd_args.no_wandb)
    
    time.sleep(60)











