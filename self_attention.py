import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, 
                 row_num: int,
                 col_num: int):
        super().__init__()

        self.d_model = d_model
        self.w_query = nn.Linear(d_model, d_model, bias=False)
        self.w_key = nn.Linear(d_model, d_model, bias=False)
        self.w_value = nn.Linear(d_model, d_model, bias=False)

        self.row_num = row_num
        self.col_num = col_num

    def forward(self, token_encodings: torch.tensor):
        q = self.w_query(token_encodings) # out_size: (3, 2)
        k = self.w_key(token_encodings)   # out_size: (3, 2)
        v = self.w_value(token_encodings) # out_size: (3, 2)

        print("================================")
        print(f"Query Size: {q.size()}")
        print(f"Key Size: {k.size()}")
        print(f"Value Size: {v.size()}")
        print("================================")

        scaled_dot_prod = (q@k.T)/torch.sqrt()


if __name__ == "__main__":
    D_MODEL = 2
    ROW_NUM = 3
    COL_NUM = 2
    input_tensor = torch.tensor([[1.16, 0.23],
                                 [0.57, 1.36],
                                 [4.41, -2.16]])
    
    self_att = SelfAttention(D_MODEL, ROW_NUM, COL_NUM)
    self_att(input_tensor)

    