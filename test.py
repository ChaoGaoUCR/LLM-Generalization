import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, src, src_mask):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class MyNetwork(nn.Module):
    def __init__(self, seq_len, embed_dim, num_heads, num_classes):
        super(MyNetwork, self).__init__()
        self.embedding = nn.Embedding(10, embed_dim)
        self.pos_encoder = nn.Embedding(seq_len, embed_dim)
        self.decoder_layer = TransformerDecoderLayer(embed_dim, num_heads)
        self.linear = nn.Linear(embed_dim, num_classes)
        self.seq_len = seq_len

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder(torch.arange(self.seq_len, dtype=torch.long, device=x.device))
        mask = self.generate_square_subsequent_mask(self.seq_len)
        x = self.decoder_layer(x, mask)
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)

seq_len = 10
model = MyNetwork(seq_len=seq_len, embed_dim=10, num_heads=2, num_classes=10)
