import torch
import torch.nn as nn
import torch.nn.functional as F
from egnn_pytorch import EGNN_Network

class CNN_Model(nn.Module):
    def __init__(self, num_residues=2, channels_1=32, kernel_size_1 = 3, channels_2=16, kernel_size_2=3, kernel_size_3=3):
        super().__init__()
        self.num_residues = num_residues
        num_features = 3 + 1 + num_residues # 3 coords, 1 mask, num_residues onehot
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=channels_1, kernel_size=kernel_size_1, padding='same')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=channels_1, out_channels=channels_2, kernel_size=kernel_size_2, padding='same')
        self.conv3 = nn.Conv1d(in_channels=channels_2, out_channels=3, kernel_size=kernel_size_3, padding='same')
    
    def forward(self, coords, residues, lengths):
        B, L, _ = coords.shape
        diff = coords[:, 1:, :] - coords[:, :-1, :]
        first_diff = torch.zeros(B, 1, 3, device=coords.device, dtype=coords.dtype)
        diff_full = torch.cat([first_diff, diff], dim=1)
        
        residues_onehot = F.one_hot(residues, num_classes=self.num_residues)

        mask = torch.arange(L, device=coords.device).unsqueeze(0).expand(B, L) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(2)
        
        x = torch.cat([diff_full, mask, residues_onehot], dim=2)
        
        # Rearrange to (batch_size, channels, sequence_length) for nn.Conv1d.
        x = x.permute(0, 2, 1)  # shape becomes (batch_size, 5, max_length)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)  # now shape: (batch_size, 3, max_length)
        # Permute back to (batch_size, max_length, 3)
        x = x.permute(0, 2, 1)
        return x
        

class Stationary_Model(nn.Module): 
    def forward(self, coords, residues, lengths):
        return torch.zeros_like(coords, device=coords.device)


class EGNN_Model(nn.Module):
    def __init__(self, max_protein_length, num_tokens=2, depth=3, emb_dim=16):
        """
        Args:
            num_tokens (int): Number of unique residue tokens.
            depth (int): Number of layers for the EGNN_Network.
            emb_dim (int): Dimension used for the network (both for embeddings and internal features).
        """
        super().__init__()
        # Here the EGNN_Network is configured to:
        # - embed the input tokens using num_tokens (if the implementation uses them internally)
        # - not expect additional edge tokens (num_edge_tokens remains None)
        # - not use extra edge features (edge_dim=0)
        # - ignore any extra adjacency degrees (num_adj_degrees=None, adj_dim=0)
        # - not apply any global linear attention (global_linear_attn_every=0)
        self.egnn = EGNN_Network(
            num_tokens=num_tokens,
            num_positions=max_protein_length,
            dim=emb_dim,
            depth=depth,
            num_nearest_neighbors=8
        )

        self.final = nn.Linear(emb_dim, 3)
        
    def create_chain_adj(self, lengths, max_length):
        """
        Create an adjacency matrix for each sample in the batch.
        For each chain, we connect residue i with residue i+1 (bidirectionally).
        
        Args:
            lengths (torch.Tensor): Tensor of shape (B,) with actual lengths.
            max_length (int): Padded chain length.
        
        Returns:
            torch.Tensor: Adjacency matrix of shape (B, max_length, max_length) with ones on edges.
        """
        B = lengths.shape[0]
        device = lengths.device
        adj = torch.zeros(B, max_length, max_length, device=device, dtype=bool)
        for b in range(B):
            L = lengths[b]
            if L > 0:
                idx = torch.arange(L - 1, device=device)
                # Connect residue i with i+1.
                adj[b, idx, idx + 1] = True
                adj[b, idx + 1, idx] = True
        return adj

    def forward(self, coords, residues, lengths):
        """
        Args:
            coords (torch.Tensor): Initial 3D coordinates with shape (B, L, 3).
            residues (torch.Tensor): Residue tokens with shape (B, L) (each token is an integer in [0, num_tokens-1]).
            lengths (torch.Tensor): Actual chain lengths with shape (B,).
        
        Returns:
            torch.Tensor: Predicted coordinate change (delta), shape (B, L, 3).
        """
        B, L, _ = coords.shape
        device = coords.device
        
        # Create mask: True for valid residues, False for padded positions.
        # This mask will tell the EGNN_Network which nodes to consider.
        mask = torch.arange(L, device=device).unsqueeze(0).expand(B, L) < lengths.unsqueeze(1)
        
        # Create adjacency matrix for chain connectivity (all samples are padded to max length L).
        adj = self.create_chain_adj(lengths, L)
        
        # Pass through EGNN_Network.
        # Here we assume that the EGNN_Network's forward method accepts keyword arguments:
        #   - coords: the node coordinates,
        #   - tokens: the residue tokens,
        #   - mask: indicating valid positions,
        #   - adj: the adjacency matrix.
        # It returns the predicted delta for each node.
        feats_out, coords_out = self.egnn(residues, coords, mask=mask, adj_mat=adj)
        
        delta = self.final(feats_out)
        return delta
