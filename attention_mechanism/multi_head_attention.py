import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):

    def __init__(self, d_model: int=2,
                 row_dim: int=0,
                 col_dim: int=1):
        super().__init__()
        
        self.w_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.w_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.row_dim = row_dim
        self.col_dim = col_dim

    
    def forward(self, encodings_for_q: torch.tensor
                ,encodings_for_k: torch.tensor
                ,encodings_for_v: torch.tensor
                ,mask=None):
        q = self.w_q(encodings_for_q)
        k = self.w_k(encodings_for_k)
        v = self.w_v(encodings_for_v)

        q_k_dot_prod = q@k.T
        scaled_dot_prod = q_k_dot_prod/torch.tensor(k.size(self.col_dim)**0.5)

        if mask is not None:
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask=mask, value=1e-5)
        
        att_weights = F.softmax(scaled_dot_prod)
        attention_scores = torch.matmul(att_weights, v)

        return attention_scores

class MultiHeadAttention(nn.Module):
    
    def __init__(self,
                 d_model: int = 2,
                 row_dim: int = 0,
                 col_dim: int = 1,
                 num_heads: int = 1):
        super().__init__()
        
        # create different attention heads
        self.heads = nn.ModuleList(
            [Attention(d_model, row_dim, col_dim)
                for _ in range(num_heads)]
        )
        
        self.col_dim = col_dim

    def forward(self
                ,encodings_for_q: torch.tensor
                ,encodings_for_k: torch.tensor
                ,encodings_for_v: torch.tensor):
        return torch.cat([head(encodings_for_q
                    ,encodings_for_k
                    ,encodings_for_v)
                for head in self.heads], dim=self.col_dim)    
    
if __name__ == "__main__":
    encodings_for_q = torch.tensor([[1.16, 0.23],
                                [0.57, 1.36],
                                [4.41, -2.16]])

    encodings_for_k = torch.tensor([[1.16, 0.23],
                                    [0.57, 1.36],
                                    [4.41, -2.16]])

    encodings_for_v = torch.tensor([[1.16, 0.23],
                                    [0.57, 1.36],
                                    [4.41, -2.16]])

    # set the seed for the random number generator
    torch.manual_seed(42)

    multi_head_attention = MultiHeadAttention(d_model=2
                                            ,row_dim=0
                                            ,col_dim=1
                                            ,num_heads=2)

    att_aware_out: torch.tensor = multi_head_attention(encodings_for_q
                                                    ,encodings_for_k
                                                    ,encodings_for_v)

    print(att_aware_out)