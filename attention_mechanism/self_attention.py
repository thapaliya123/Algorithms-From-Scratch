"""
Implement self attention block of Attention is all you Need Paper:
Formuale: 
Scaled Dot product Attention = Softmax(Q*K/sqrt(dk))*V
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, 
                 row_dim: int = 0,
                 col_dim: int = 1):
        super().__init__()

        self.d_model = d_model
        self.w_query = nn.Linear(d_model, d_model, bias=False)
        self.w_key = nn.Linear(d_model, d_model, bias=False)
        self.w_value = nn.Linear(d_model, d_model, bias=False)

        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, token_encodings: torch.tensor) -> torch.tensor:
        q = self.w_query(token_encodings) # out_size: (3, 2)
        k = self.w_key(token_encodings)   # out_size: (3, 2)
        v = self.w_value(token_encodings) # out_size: (3, 2)

        print("================================")
        print(f"Query Size: {q.size()}")
        print(f"Key Size: {k.size()}")
        print(f"Value Size: {v.size()}")
        print("================================")

        q_k_dot_prod = q@k.T # out shape: (3, 3)
        scaled_dot_prod = q_k_dot_prod/torch.tensor(k.size(self.col_dim)**0.5) # out_shape: (3, 3)
        att_weights = F.softmax(scaled_dot_prod, dim=self.col_dim)
        att_scores = att_weights@v
        
        return att_scores

if __name__ == "__main__":
    D_MODEL = 2
    ROW_NUM = 0
    COL_NUM = 1
    torch.manual_seed(42)
    input_tensor = torch.tensor([[1.16, 0.23],
                                 [0.57, 1.36],
                                 [4.41, -2.16]])
    
    self_att = SelfAttention(D_MODEL, ROW_NUM, COL_NUM)
    att_scores = self_att(input_tensor)

    print(f"Attention Scores:\n{att_scores}")
    