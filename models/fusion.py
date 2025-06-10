import torch
from torch import nn
from einops import rearrange

class Mix_TR(nn.Module):
    def __init__(self, num_writers, writer_emb_dim, content_encoder, decoder):
        super().__init__()
        self.writer_embedding = nn.Embedding(num_writers, writer_emb_dim)
        self.content_encoder = content_encoder
        self.decoder = decoder

    def forward(self, writer_ids, content):
        # writer_ids: (batch,)
        writer_emb = self.writer_embedding(writer_ids)  # (batch, emb_dim)
        # Expand writer_emb to match content dims if necessary
        
        content = rearrange(content, 'n t h w -> (n t) 1 h w').contiguous()
        content = self.content_encoder(content)
        content = rearrange(content, '(n t) c h w -> t n (c h w)', n=writer_ids.shape[0]).contiguous()
        
        # Optionally, combine writer_emb with content encoding as needed by your model
        # Example: concat or add
        # Here we expand writer_emb to (t, n, emb_dim) and concat to content
        t = content.shape[0]
        writer_emb_exp = writer_emb.unsqueeze(0).expand(t, -1, -1)
        combined = torch.cat([content, writer_emb_exp], dim=-1)
        
        output = self.decoder(combined)
        return output
