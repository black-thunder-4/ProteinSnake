import os
import torch
from torch.utils.data import Dataset, DataLoader
import mdtraj as md
from tqdm import tqdm
import numpy as np

def get_pdb_files(path):
    """
    Recursively collects all PDB files if a directory is provided, or returns
    the single file if a PDB file is provided.
    """
    if os.path.isfile(path):
        if path.endswith('.pdb'):
            return [path]
        else:
            raise ValueError(f"File {path} is not a pdb file.")
    elif os.path.isdir(path):
        pdb_files = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.pdb'):
                    pdb_files.append(os.path.join(root, file))
        return sorted(pdb_files)
    else:
        raise ValueError(f"Path {path} is neither a valid file nor directory.")

class ProteinTrajectoryDataset(Dataset):

    def __init__(self, path, n_steps=10, nr=2):
        """
        Args:
            n_steps (int): Fixed number of steps in the future to compute the change.
            transform (callable, optional): Optional transform to be applied on a sample.
            nr (int): Number of residue types to consider (e.g. 2 for ALA and GLY)
        """
        self.n_steps = n_steps
        self.nr = nr
        self.trajectories = [] # (n_frames, n_atoms, 3 + nr)
        self.residues = [] # (n_atoms)
        self.indices = [] # List of tuples: (trajectory_index, frame_index)
    
        pdb_files = get_pdb_files(path)
        
        mapping = {'A': 0, 'G': 1}
        
        # Load all trajectories with a progress bar.
        for file in tqdm(pdb_files, desc="Loading trajectories"):
            
            traj = md.load(file)
            
            positions = traj.xyz  # shape: (n_frames, n_atoms, 3)
            self.trajectories.append(positions)
            
            residues = [mapping[res.code] for res in traj.top.residues]
            residues = np.array(residues)
    
            self.residues.append(residues)
            
            # Only add indices for frames that have a valid future frame (n_steps ahead)
            for i in range(positions.shape[0] - self.n_steps):
                self.indices.append((len(self.trajectories) - 1, i))
                    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        traj_idx, frame_idx = self.indices[idx]
        traj = self.trajectories[traj_idx]
        current_coords = traj[frame_idx]                        # shape: (n_atoms, 3)
        future_coords = traj[frame_idx + self.n_steps]          # shape: (n_atoms, 3)
        delta = future_coords - current_coords

        sample = {'coords': current_coords, 'residues': self.residues[traj_idx], 'delta': delta}
        
        # Convert the numpy arrays to torch tensors.
        sample['coords'] = torch.tensor(sample['coords'], dtype=torch.float)
        sample['residues'] = torch.tensor(sample['residues'], dtype=torch.long)
        sample['delta'] = torch.tensor(sample['delta'], dtype=torch.float)
        
        return sample