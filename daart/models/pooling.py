import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import math


class TC(nn.Module):
    def __init__(self, cls_dim: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(cls_dim, cls_dim, kernel_size, padding=padding)
        self.norm = nn.LayerNorm(cls_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, seq_len, cls_dim)
        x_orig = x

        x = x.transpose(1, 2)  # (batch_size, cls_dim, seq_len)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, cls_dim)

        x = self.relu(x)
        x = self.dropout(x)
        x = x + x_orig  # residual
        x = self.norm(x)

        return x


class RE(nn.Module):
    def __init__(self, max_rel_dist, embed_dim):
        """
        max_rel_dist: Maximum distance to consider for relative position
        embed_dim: Dimension of CLS token embeddings
        """
        super().__init__()
        self.max_rel_dist = max_rel_dist
        self.embed_dim = embed_dim
        self.rel_pos_embed = nn.Embedding(2 * max_rel_dist + 1, embed_dim)

    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.size()
        device = x.device

        # Compute relative positions from center
        center = seq_len // 2
        rel_positions = torch.arange(seq_len, device=device) - center
        rel_positions = rel_positions.clamp(-self.max_rel_dist, self.max_rel_dist) + self.max_rel_dist
        bias = self.rel_pos_embed(rel_positions)  # (seq_len, embed_dim)
        bias = bias.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_len, embed_dim)

        return x + bias


class FPE(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].to(x.device)

class TW(nn.Module):
    def __init__(self, seq_len, cls_dim,  n_heads=4, n_layers=2):
        super().__init__()
        self.cls_dim = cls_dim
        self.seq_len = seq_len
        
        # Transformer encoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=cls_dim, 
            nhead=n_heads, 
            dim_feedforward=cls_dim * 2
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=n_layers
        )
        
        # LayerNorm to stabilize features
        self.layer_norm = nn.LayerNorm(cls_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, cls_dim)
        Returns:
            Tensor of shape (batch_size, seq_len, cls_dim)
        """
        # Ensure input is in the shape (seq_len, batch_size, cls_dim) for Transformer
        x = x.transpose(0, 1)  # (seq_len, batch_size, cls_dim)
        
        # Apply the transformer encoder to the sequence of CLS tokens
        transformed_x = self.transformer_encoder(x)  # (seq_len, batch_size, cls_dim)
        
        # Add residual connection (original input + transformed output)
        residual_output = transformed_x + x  # (seq_len, batch_size, cls_dim)
        
        # Apply LayerNorm to the residual output
        #normalized_x = self.layer_norm(residual_output)  # (seq_len, batch_size, cls_dim)
        
        # Transpose back to (batch_size, seq_len, cls_dim)
        #return normalized_x.transpose(0, 1)  # (batch_size, seq_len, cls_dim)
        return residual_output.transpose(0, 1)

class AddPositionalEncoding(nn.Module):
    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, dim))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, dim)
        Returns:
            Tensor of shape (batch_size, seq_len, dim)
        """
        return x + self.pos_embed
        
class MAP(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.norm(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)  # (B, T, C)
        return x + self.dropout(attn_output)

class MAPF(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Attention block
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.dropout1(attn_output)

        # Feedforward block
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)

        return x



class LinearDimReducer(nn.Module):
    def __init__(self, in_dim=768, out_dim=256):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, in_dim)
        Returns:
            Tensor of shape (batch_size, seq_len, out_dim)
        """
        return nn.ReLU()(self.proj(x))

class PMA(nn.Module):
    def __init__(self, in_dim, num_heads, d, out_dim, dropout_rate=0.1, use_positional_encoding=True):
        """
        Initializes the Pool of Multi-Head Attention (PMA) module with optional positional encoding.
        
        Args:
        - in_dim (int): Input dimension of the sequence
        - num_heads (int): Number of attention heads
        - d (int): Dimensionality of each attention head
        - out_dim (int): Output dimension of the pooled sequence
        - dropout_rate (float): Dropout rate for regularization
        - use_positional_encoding (bool): Whether to add positional encoding
        """
        super(PMA, self).__init__()
        self.num_heads = num_heads
        self.d = d
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.use_positional_encoding = use_positional_encoding

        # Linear projections for queries, keys, and values
        self.fc_q = nn.Linear(in_dim, num_heads * d)
        self.fc_k = nn.Linear(in_dim, num_heads * d)
        self.fc_v = nn.Linear(in_dim, num_heads * d)

        # Output projection to combine the attention heads into out_dim
        self.fc_out = nn.Linear(num_heads * d, out_dim)

        # Dropout layer
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # Positional encoding
        if self.use_positional_encoding:
            self.positional_encoding = nn.Parameter(torch.randn(1, 1, in_dim))

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, X):
        """
        Forward pass for the PMA layer.
        
        Args:
        - X (Tensor): Input sequence with shape (B, T, in_dim)
        
        Returns:
        - output (Tensor): Output sequence with shape (B, T, out_dim)
        """
        B, T, in_dim = X.shape  # X has shape (B, T, in_dim)

        # Optionally add positional encoding
        if self.use_positional_encoding:
            X = X + self.positional_encoding

        # Ensure that in_dim is divisible by num_heads
        assert in_dim % self.num_heads == 0

        # Project input into queries, keys, and values
        Q = self.fc_q(X).view(B, T, self.num_heads, self.d).transpose(1, 2)  # (B, num_heads, T, d)
        K = self.fc_k(X).view(B, T, self.num_heads, self.d).transpose(1, 2)  # (B, num_heads, T, d)
        V = self.fc_v(X).view(B, T, self.num_heads, self.d).transpose(1, 2)  # (B, num_heads, T, d)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d)  # (B, num_heads, T, T)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)  # (B, num_heads, T, T)
        attention_output = torch.matmul(attention_weights, V)  # (B, num_heads, T, d)

        # Combine the attention outputs from all heads (reshape to (B, T, num_heads * d))
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.d)  # (B, T, num_heads * d)

        # Apply dropout for regularization
        attention_output = self.dropout(attention_output)

        # Final linear layer to project to out_dim
        output = self.fc_out(attention_output)  # (B, T, out_dim)

        # Layer normalization
        output = self.layer_norm(output)

        return output


class MBA(nn.Module):
    def __init__(self, in_dim, num_heads, d, out_dim, dropout_rate=0.1, use_positional_encoding=True):
        """
        A model integrating the PMA mechanism for pooling over sequences.
        
        Args:
        - in_dim (int): Input dimension of the sequence
        - num_heads (int): Number of attention heads
        - d (int): Dimensionality of each attention head
        - out_dim (int): Output dimension of the pooled sequence
        - dropout_rate (float): Dropout rate for regularization
        - use_positional_encoding (bool): Whether to add positional encoding
        """
        super(MBA, self).__init__()
        
        # Define the PMA layer
        self.pma = PMA(in_dim, num_heads, d, out_dim, dropout_rate, use_positional_encoding)

    def forward(self, X):
        """
        Forward pass through the model.
        
        Args:
        - X (Tensor): Input sequence with shape (B, T, in_dim)
        
        Returns:
        - output (Tensor): Output sequence with shape (B, T, out_dim)
        """
        return self.pma(X)




# # daart/models/pooling.py

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MAB(nn.Module):
#     def __init__(self, dim, num_heads, ln=False):
#         super().__init__()
#         self.dim_V = dim
#         self.num_heads = num_heads
#         self.fc_q = nn.Linear(dim, dim)
#         self.fc_k = nn.Linear(dim, dim)
#         self.fc_v = nn.Linear(dim, dim)
#         if ln:
#             self.ln0 = nn.LayerNorm(dim)
#             self.ln1 = nn.LayerNorm(dim)
#         else:
#             self.ln0 = None
#             self.ln1 = None
#         self.fc_o = nn.Linear(dim, dim)

#     def forward(self, Q, K):
#         # Q, K: (B, S, dim)
#         B, SQ, D = Q.shape
#         _, SK, _ = K.shape
#         d = D // self.num_heads

#         Q = self.fc_q(Q)
#         K = self.fc_k(K)
#         V = self.fc_v(K)

#         Q = Q.view(B, SQ, self.num_heads, d).transpose(1, 2)  # (B, num_heads, SQ, d)
#         K = K.view(B, SK, self.num_heads, d).transpose(1, 2)
#         V = V.view(B, SK, self.num_heads, d).transpose(1, 2)

#         A = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(d), dim=-1)
#         O = (A @ V).transpose(1, 2).reshape(B, SQ, D)

#         O = Q.transpose(1, 2).reshape(B, SQ, D) + O  # Residual from Q input

#         if self.ln0 is not None:
#             O = self.ln0(O)

#         O = O + F.relu(self.fc_o(O))

#         if self.ln1 is not None:
#             O = self.ln1(O)

#         return O

# class PMA(nn.Module):
#     def __init__(self, dim, num_heads, num_seeds, ln=False):
#         super().__init__()
#         self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
#         nn.init.xavier_uniform_(self.S)
#         self.mab = MAB(dim, num_heads, ln=ln)

#     def forward(self, X):
#         B = X.size(0)
#         return self.mab(self.S.repeat(B, 1, 1), X)
