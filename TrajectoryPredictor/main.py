from ProteinTrajectoryDataset import *
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import tqdm
import optuna
from Models import *
import argparse
import time

# CLI for GPU selection and Optuna storage URL
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--train_set_path', type=str, required=True)
parser.add_argument('--val_set_path', type=str, required=True)
parser.add_argument('--storage_path', type=str, default='optuna_study.db')
parser.add_argument('--study_name', type=str, requited=True)
parser.add_argument('--save_dir', type=str, required=True,
                    help='Directory to save model state dicts for each Optuna trial')
parser.add_argument('--n_trials', type=int, default=50)
parser.add_argument('--timeout_per_trial', type=int, default=30*60)
args = parser.parse_args()

# Restrict visible GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MAX_PROTEIN_LENGTH = 50

def collate_fn(batch):
    coords = [item['coords'] for item in batch]
    residues = [item['residues'] for item in batch]
    deltas = [item['delta'] for item in batch]
    lengths = torch.tensor([c.shape[0] for c in coords])
    # Pad sequences to the maximum length in the batch (batch_first=True gives shape [batch, max_length, features])
    coords = pad_sequence(coords, batch_first=True)
    residues = pad_sequence(residues, batch_first=True)
    deltas = pad_sequence(deltas, batch_first=True)
    return coords, residues, deltas, lengths

def masked_mse_loss(pred, target, lengths):
    batch_size, max_length, _ = pred.shape
    # Create a mask with shape (batch_size, max_length) where each element is True if it is a valid timestep.
    mask = torch.arange(max_length, device=pred.device).expand(batch_size, max_length) < lengths.unsqueeze(1)
    # Expand mask to match the last dimension (3) of our tensors.
    mask = mask.unsqueeze(2).float()  # shape becomes (batch_size, max_length, 1)
    
    # Compute squared differences
    mse = (pred - target) ** 2
    # Zero-out padded elements using the mask and compute average loss only over valid elements.
    loss = (mse * mask).sum() / mask.sum()
    return loss

def build_model(trial):
    match args.model:
        case "cnn":
            return CNN_Model(
                channels_1=trial.suggest_categorical('channels_1', [4, 8, 16, 32, 64, 128]),
                channels_2=trial.suggest_categorical('channels_2', [4, 8, 16, 32, 64, 128]),
                kernel_size_1=trial.suggest_categorical('kernel_size_1', [1, 3, 5, 7]),
                kernel_size_2=trial.suggest_categorical('kernel_size_2', [1, 3, 5, 7]),
                kernel_size_3=trial.suggest_categorical('kernel_size_3', [1, 3, 5, 7]),
            )
        case "egnn":
            return EGNN_Model(
                MAX_PROTEIN_LENGTH, 
                depth=trial.suggest_int('depth', 1, 4),
                emb_dim=trial.suggest_categorical('emb_dim', [4, 8, 16, 32, 64, 128, 256, 512, 1024])
            )
        case _:
            raise ValueError(f"Unknown model: {args.model}")


def objective(trial):
    # Sample hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256, 512, 1024])

    # Prepare data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Build model
    model = build_model(trial)
    
    # Wrap with DataParallel if multiple GPUs are specified
    #if len(gpu_ids) > 1:
    #    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)

    # Choose optimizer
    params = model.parameters()
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)

    coordinate_factor = trial.suggest_float('coordinate_factor', 0.1, 10, log=True)

    start_time = time.time()

    # Training loop
    max_epochs = 100  # Set a reasonable upper limit to avoid infinite loops
    epoch = 0
    while epoch < max_epochs:
        model.train()
        for coords, residues, deltas, lengths in tqdm.tqdm(train_loader):
            coords = coords.to(device) * coordinate_factor
            residues = residues.to(device)
            deltas = deltas.to(device) * coordinate_factor
            lengths = lengths.to(device)

            optimizer.zero_grad()
            pred_deltas = model(coords, residues, lengths)
            loss = masked_mse_loss(pred_deltas, deltas, lengths)
            loss.backward()
            optimizer.step()

        # Intermediate evaluation
        running_loss = 0
        total_samples = 0
        model.eval()
        for coords, residues, deltas, lengths in tqdm.tqdm(val_loader):
            coords = coords.to(device) * coordinate_factor
            residues = residues.to(device)
            deltas = deltas.to(device) * coordinate_factor
            lengths = lengths.to(device)

            pred_deltas = model(coords, residues, lengths)
            loss = masked_mse_loss(pred_deltas, deltas, lengths)

            batch_size = coords.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
        
        val_loss = running_loss / total_samples / (coordinate_factor ** 2)
        trial.report(val_loss, epoch)

        # Optuna pruning
        if trial.should_prune() or (time.time() - start_time) > args.timeout_per_trial:
            raise optuna.TrialPruned()

        epoch += 1
        
    # Save model state dict for this completed trial
    save_path = os.path.join(args.save_dir, f"model_trial_{trial.number}.pth")
    torch.save(model.state_dict(), save_path)

    return val_loss

if __name__ == "__main__":

    train_dataset = ProteinTrajectoryDataset(args.train_set_path, n_steps=1)
    val_dataset = ProteinTrajectoryDataset(args.val_set_path, n_steps=1)

    storage_url = f"sqlite:///{args.storage_path}"
    
    # Create or load study with persistent storage and pruning
    study = optuna.create_study(
        study_name='optuna_hpo',
        direction='minimize',
        storage=storage_url,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )
    study.optimize(objective, n_trials=args.n_trials)