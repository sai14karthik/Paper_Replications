import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
       
        super().__init__()
        self.d_model = d_model
        self.vocab_size=vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        # Scale embeddings by sqrt(d_model) as in the paper
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1):
       
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)
        self.register_buffer('pe', pe)
        
    
    def forward(self, x):
        x=x+(self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
        

class LayerNormalization(nn.Module):
    
    def __init__(self, eps:float=10**-6) ->None :
        
        super().__init__()
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(1)) 
        self.bias=nn.Parameter(torch.zeros(1))
        
    def forward(self,x):
        
        mean=x.mean(dim=-1, keepdim=True)
        std=x.std(dim=-1, keepdim=True)
        
        return self.alpha*(x-mean)/(std+self.eps)+self.bias
        
        

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None :
        super().__init__()
        
        self.linear_1=nn.Linear(d_model,d_ff)
        self.dropout=nn.Dropout(dropout)
        self.linear_2=nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        

