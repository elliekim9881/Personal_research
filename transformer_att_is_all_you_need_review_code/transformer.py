import torch
import torch.nn as nn

device = torch.device("mps")
BATCH_SIZE = 128

# Multi-Head Attention

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()
        
        assert hidden_dim % self.n_heads == 0
        
        self.hidden_dim = hidden_dim #embedding dim
        self.n_heads = n_heads # head num
        self.hidden_dim = hidden_dim // n_heads # each head embed dim

        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout_ratio)

        self.scale = torch.sqrt(torch.FloatTensor([self.hidden_dim])).to(device)

    def split_heads(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q= self.fc_q(query)
        K= self.fc_k(key)
        V= self.fc_v(value)
        
        #n_head(h)개의 서로 다른 att 컨셉 학습 유도
        Q = Q.view(batch_size, -1, self.n_heads, self.hidden_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.hidden_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.hidden_dim).permute(0, 2, 1, 3)

        #att energy compute
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        # masking
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        #att score compute
        attention = torch.softmax(energy, dim=-1)
        
        #scaled dot porduct att
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hidden_dim)
        x = self.fc_o(x)
        
        return x, attention

# position wise feed forward network
class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, pf_dim, dropout_ratio, device):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
    
# encoder layer
class encoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiHeadAttention(hidden_dim, n_heads, dropout_ratio, device)
        self.positionwise_feedforward = PositionWiseFeedForward(hidden_dim, pf_dim, dropout_ratio, device)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x, mask):
        _x, _ = self.self_attention(x, x, x, mask)
        x = self.self_attn_layer_norm(x + self.dropout(_x))

        _x = self.positionwise_feedforward(x)
        x = self.ff_layer_norm(x + self.dropout(_x))

        return x


# encoder
class encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_heads, pf_dim, dropout_ratio, device, max_length=100):
        super().__init__()

        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)

        self.layers = nn.ModuleList([encoderLayer(hidden_dim, n_heads, pf_dim, dropout_ratio, device) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout_ratio)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        
# decoder layer
class decoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiHeadAttention(hidden_dim, n_heads, dropout_ratio, device)
        self.encoder_attention = MultiHeadAttention(hidden_dim, n_heads, dropout_ratio, device)
        self.positionwise_feedforward = PositionWiseFeedForward(hidden_dim, pf_dim, dropout_ratio, device)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        


# decoder
class decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers, n_heads, pf_dim, dropout_ratio, device, max_length=100):
        super().__init__()

        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)

        self.layers = nn.ModuleList([decoderLayer(hidden_dim, n_heads, pf_dim, dropout_ratio, device) for _ in range(n_layers)])

        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout_ratio)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        
#transformer
class transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output, attention