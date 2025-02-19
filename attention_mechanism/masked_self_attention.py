"""
Implement masked self attention block of Attention is all you Need Paper:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model: int = 2,
                 row_dim: int = 0,
                 col_dim: int = 1):
        super().__init__()
        self.d_model = d_model
        self.w_query = nn.Linear(d_model, d_model, bias=False)
        self.w_key = nn.Linear(d_model, d_model, bias=False)
        self.w_value = nn.Linear(d_model, d_model, bias=False)

        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, encodings: torch.tensor, 
                mask: torch.tensor = None):
        q = self.w_query(encodings)
        k = self.w_key(encodings)
        v = self.w_value(encodings)

        q_k_dot_prod = q@k.T
        scaled_dot_prod = q_k_dot_prod/torch.tensor(k.size(self.col_dim)**0.5)

        if mask is not None:
            # mask out things that doesnot requires to pay attention
            # we replace values we wanted masked out
            # with a very small negative number so that the SoftMax() function
            # Will give all masked elements an output value (or "probability") of 0.
            # breakpoint()
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask=mask, value=1e-5)
        
        att_weights = F.softmax(scaled_dot_prod, dim=self.col_dim)
        attention_scores = torch.matmul(att_weights, v)

        return attention_scores



if __name__ == "__main__":
    # create a matrix of token encodings.....
    encoding_matrix = torch.tensor([[1.16, 0.23],
                                    [0.57, 1.36],
                                    [4.41, -2.16]])

    # set the seed for the random number generator
    torch.manual_seed(42)

    # create a masked self-attention object
    masked_self_attention = MaskedSelfAttention(d_model=2,
                                                row_dim=0,
                                                col_dim=1)


    mask = torch.tril(torch.ones(3, 3))
    mask = mask == 0

    out = masked_self_attention(encoding_matrix, mask)
    print(out)