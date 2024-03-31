import torch
import torch.nn as nn

#device = torch.device("mps")
BATCH_SIZE = 128

# Multi-Head Attention
# 쿼리, 키, 밸류 (차원 모두 동일)
# 하이퍼파라미터
# hidden_dim 하나의 단어에 대한 임베딩 차원
# n_head  헤드의 개수 = scaled dot product attention의 개수
# propout_rate 드롭아웃 비율

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()
        
        assert hidden_dim % self.n_heads == 0
        
        self.hidden_dim = hidden_dim #embedding dim
        self.n_heads = n_heads # head num
        self.head_dim = hidden_dim // n_heads # each head embed dim

        # 임베딩 차원에서 키의 차원으로 바꾼다 > 실제구현 : hidden_dim to hidden_dim 매핑 후 n_head로 쪼갬
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout_ratio)

        # Q K V 에 루트를 씌운 값을 나누어줌, softmax에 넣어줄 수 있도록
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def split_heads(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q= self.fc_q(query)
        K= self.fc_k(key)
        V= self.fc_v(value)
        
        #n_head(h)개의 서로 다른 att 컨셉 학습 유도
        #linear를 각각 거친 값을 구할 수 있다
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        #att energy compute
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        # masking 사용할 경우에
        if mask is not None:
            # 마스크값이 0인 부분 무한에 가까운 값으로 (-1e10), softmax > 0 에 가깝도록 만듬
            energy = energy.masked_fill(mask == 0, -1e10)
        
        #att score compute
        attention = torch.softmax(energy, dim=-1)
        
        #scaled dot porduct att
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        # concat 수행한것과 동일한 결과
        x = x.view(batch_size, -1, self.hidden_dim)
        #output linear 거침
        x = self.fc_o(x)
        
        return x, attention

# position wise feed forward network
class PositionWiseFeedForward(nn.Module):
    #입력과 출력의 차원이 동일
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
# attention - residual connection, normalization - feedforward network - residual connection,normalization   
class encoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiHeadAttention(hidden_dim, n_heads, dropout_ratio, device)
        self.positionwise_feedforward = PositionWiseFeedForward(hidden_dim, pf_dim, dropout_ratio, device)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x, mask):
        # src, src,src = Q, K, V(복제되어 같은 값 들어감), mask 특정단어에 대해서는 어텐션 수행 x
        _src, _ = self.self_attention(src, src, src, mask)
        src = self.self_attn_layer_norm(x + self.dropout(_src))

        _src = self.positionwise_feedforward(x)
        src = self.ff_layer_norm(src + self.dropout(_src))

        return src


# encoder
class encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_heads, pf_dim, dropout_ratio, device, max_length=100):
        super().__init__()

        #실제 단어의 개수 : input_dim
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hidden_dim)
        #위치 임베딩 학습하는 형태
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)

        # n_layer 만큼 반복
        self.layers = nn.ModuleList([encoderLayer(hidden_dim, n_heads, pf_dim, dropout_ratio, device) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout_ratio)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(self, src, src_mask):
        batch_size = src.shape[0] #문장의 개수
        src_len = src.shape[1] #단어의 개수가 가장 많은 단어의 개수

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        # 입력값 = 문장의 임베딩 + 위치 임베딩
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        #인코더 레이어 모두 거침(forward 수행)
        for layer in self.layers:
            src = layer(src, src_mask)
        
        return src
        
# decoder layer
# 두개의 multi-head attention레이어 사용 : self, e-d attention
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
        # Q, K, V 복제해서 입력, 자기 자신에 대해서 attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        # dropout, residual connection, layer normalization
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # 디코더의 쿼리 사용, 인코더에서 가장 마지막 값으로 나온 값을 K로 사용, attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        
        _trg = self.positionwise_feedforward(trg)
        
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        return trg, attention


# decoder
# 타겟이 되는 문장에서 다음 단어가 무엇인지 알 수 없도록 만들기 위하여 마스크벡터사용
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
        
        #0~문장길이하나의 텐서로 초기화, 각 문장에 대해서 동일하게 적용
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #실제입력
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        output = self.fc_out(trg)
        
        return output, attention
        
#transformer
class transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    # 입력 문장에 따라서 마스크 적용 / <pad> 토큰에 대해서 mask 0으로 설정
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask
    # 타겟 문장에서 각 단어가 다음 단어가 무엇인지 알 수 없도록
    def make_trg_mask(self, trg):
        
        # 1 0 0 0 0
        # 1 1 0 0 0
        # 1 1 1 0 0
        # 1 1 1 0 0
        # 1 1 1 0 0
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        trg_len = trg.shape[1]
        #앞쪽의 단어만 볼 수 있도록하는 별도의 마스크
        # 1 0 0 0 0
        # 1 1 0 0 0
        # 1 1 1 0 0 
        # 1 1 1 1 0
        # 1 1 1 1 1
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # 둘다 1인 경우에만 attention score 구하도록
        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask
    
    def forward(self, src, trg):
        #마스크 생성
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #인코더 출력값 생성
        enc_src = self.encoder(src, src_mask)

        #디코더에서 인코더의 출력값을 어텐션 한다
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output이 최종 결과
        return output, attention