from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# MODEL CONFIG PARAMETERS
@dataclass
class ModelDimensions: 
    n_mels: int = None  # Number of audio mel-frequency bins expected.
    n_vocab: int =  None  # Size of whisper Vocabulary.
    n_audio_ctx: int = None  # Number of frames(context length) in the encoder output representation.
    n_audio_state: int = None  # Dimensionality of each frame of the encoder output representation.
    n_audio_head: int = None  # Number of heads in the audio encoder multi-head self-attention layers.
    n_audio_layer: int = None  # Number of blocks in the encoder.
    n_text_ctx: int = None  # Max number of tokens to be used as a context in the decoder.
    n_text_state: int = None  # Dimensionality of each token embedding.
    n_text_head: int = None # Number of heads in the text decoder multi-head attention layers.
    n_text_layer: int = None  # Number of blocks in the decoder.


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        
        self.stride = stride
        self.padding = padding
        # The weights should have 384 filters each of shape (in_channels=80, kernel_size=3)
        self.weight = nn.Parameter(torch.randn((out_channels, in_channels, kernel_size)))
        self.bias = nn.Parameter(torch.randn((out_channels))) # one bias for each channel.

    def forward(self, x):
        out = F.conv1d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding)
        return out
    

class GELU(nn.Module):
    """Gelu is an activation functions that combines the functionality of relu and dropout.
    
    ReLu deterministically multiplies values less than zero by zero (dropping) and multiplies
    values greater than zero by 1.
    Dropout stochastically multiplies each value with a zero or one with a given probability. But
    it does not care about its size.
    Gelu merges the functionality of the two by multiplying an input value by a probability
    that depends on its size. For a very small value the probability is very close to zero so
    the value is clipped but for very large value the probability is very close to one.
    """
    def __init__(self):
        super().__init__()
        self.distribution = torch.distributions.normal.Normal(0, 1)
        
    def forward(self, x):
        x = x * self.distribution.cdf(x)
        return x
    

class Embedding(nn.Module):
    def __init__(self, n_vocab, n_state):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((n_vocab, n_state)))
        
    def forward(self, x):
        x = self.weight[x]
        return x


class LayerNorm(nn.Module):
    """Normalizes the activations of each vector in a given sequence independently."""
    def __init__(self, n_state, eps=1e-05):
        super().__init__()
        
        self.eps = eps
        self.n_state = n_state
        self.weight = nn.Parameter(torch.ones((n_state,)))
        self.bias = nn.Parameter(torch.zeros((n_state,)))
        
    def forward(self, x):
        mean = x.mean(dim=2, keepdim=True)
        n_state = x.shape[-1]
        variance = torch.sum((x - mean)**2, dim=2, keepdim=True) / n_state
        out = (x - mean) / torch.sqrt(variance + self.eps)
        out = out * self.weight + self.bias
        return out


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn((out_features, in_features)))
        self.bias = nn.Parameter(torch.randn((out_features,))) if bias else None
        
    def forward(self, x):
        x = x @ self.weight.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, n_state):
        super().__init__()
        
        self.n_head = n_head
        self.n_state = n_state
        self.d_head = n_state // n_head
        self.query = Linear(n_state, n_head * self.d_head)
        self.key = Linear(n_state, n_head * self.d_head, bias=False)
        self.value = Linear(n_state, n_head * self.d_head)
        self.out = Linear(n_head * self.d_head, n_state)
        
    def forward(self, x, xa=None, mask=None):
        q = self.query(x)
        if xa is None:
            k = self.key(x)
            v = self.value(x)
        else:
            k = self.key(xa)
            v = self.value(xa)
        qkv = self._qkv_attention(q, k, v, mask)
        out = self.out(qkv)
        return out
    
    def _qkv_attention(self, q, k, v, mask=None):
        n_batch, q_ctx = q.shape[0], q.shape[1]
        kv_ctx = k.shape[1]
        scale = self.d_head ** -0.25
        q = q.view(n_batch, q_ctx, self.n_head, self.d_head).permute(0, 2, 1, 3) * scale # (1, n_head, T, d_head)
        k = k.view(n_batch, kv_ctx, self.n_head, self.d_head).permute(0, 2, 3, 1) * scale # (1, n_head, d_head, T)
        v = v.view(n_batch, kv_ctx, self.n_head, self.d_head).permute(0, 2, 1, 3) # (1, n_head, T, d_head)
        qk = q @ k # (1, n_head, T1, T1)
        if mask is not None:
            qk += mask[:q_ctx, :kv_ctx] # Slice the rows and columns you need.
        qk = F.softmax(qk, dim=-1)
        qkv = qk @ v # -> (1, n_head, T, d_head)
        qkv = qkv.permute(0, 2, 1, 3).flatten(start_dim=2)
        return qkv

    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state, n_head, n_mlp, cross_attention=False):
        super().__init__()
        
        self.attn = MultiHeadAttention(n_head, n_state)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_head, n_state) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        self.mlp = nn.Sequential(Linear(n_state, n_mlp), GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(self, x, xa=None, mask=None):
        x = x + self.attn(self.attn_ln(x), mask=mask)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)
        x = x + self.mlp(self.mlp_ln(x))
        return x
    
    
class AudioEncoder(nn.Module):
    def __init__(self, n_mels, n_audio_layer, n_audio_ctx, n_audio_state, n_audio_head):
        super().__init__()
        
        self.conv1 = Conv1d(n_mels, n_audio_state, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv1d(n_audio_state, n_audio_state, kernel_size=3, stride=2, padding=1)
        self.gelu = GELU()
        self.register_buffer('positional_embedding', self._get_pos_encoding(n_audio_ctx, n_audio_state))

        n_audio_mlp = n_audio_state * 4
        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_audio_state, n_audio_head, n_audio_mlp) for _ in range(n_audio_layer)]
        )
        self.ln_post = LayerNorm(n_audio_state)

    def forward(self, x):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, f"incorrect audio shape {x.shape}"
        x += self.positional_embedding

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x
    
    def _get_pos_encoding(self, n_audio_ctx, n_audio_state):
        # We first operate on the dim mask where we compute the encodings that depend only
        # on the dimension. It has shape (1, n_state//2)
        dim_mask = torch.arange(n_audio_state//2).view(1, -1)
        factor = torch.log(torch.tensor(10_000)) / (n_audio_state // 2 - 1)
        dim_mask = torch.exp(-factor * dim_mask)
        # A position mask for every position in the context. It has shape (n_ctx, 1)
        pos_mask = torch.arange(n_audio_ctx).view(n_audio_ctx, 1)
        # Here, when we multiply the masks, both masks gets broadcasted to shape (n_ctx, n_state//2)
        # which is also the output shape.
        mask = pos_mask * dim_mask
        # Concatenate to get shape (n_ctx, n_state)
        pos_encoding = torch.cat((torch.sin(mask), torch.cos(mask)), dim=1)
        return pos_encoding
    

class TextDecoder(nn.Module):
    def __init__(self, n_vocab, n_text_layer, n_text_ctx, n_text_state, n_text_head):
        super().__init__()
        
        self.token_embedding = Embedding(n_vocab, n_text_state)
        # Learned pos embedding.
        self.positional_embedding = nn.Parameter(torch.empty(n_text_ctx, n_text_state))
        n_text_mlp = n_text_state * 4
        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_text_state, n_text_head, n_text_mlp, cross_attention=True)
            for _ in range(n_text_layer)]
        )
        self.ln = LayerNorm(n_text_state)

        mask = torch.full((n_text_ctx, n_text_ctx), float('-Infinity')).triu_(diagonal=1)
        # persistent argument ensures the mask is not included in the state dict of the module.
        self.register_buffer('mask', mask, persistent=False)

    def forward(self, x, xa):
        x = self.token_embedding(x) + self.positional_embedding[: x.shape[-1]]
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return logits
    

class Whisper(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.encoder = AudioEncoder(
            n_mels=dims.n_mels,
            n_audio_layer=dims.n_audio_layer,
            n_audio_ctx=dims.n_audio_ctx, 
            n_audio_state=dims.n_audio_state,
            n_audio_head=dims.n_audio_head, 
        )
        self.decoder = TextDecoder(
            n_vocab=dims.n_vocab, 
            n_text_layer=dims.n_text_layer, 
            n_text_ctx=dims.n_text_ctx,
            n_text_state=dims.n_text_state, 
            n_text_head=dims.n_text_head,
        )
    
    @torch.no_grad()
    def embed_audio(self, mel):
        return self.encoder.forward(mel)
    
    @torch.no_grad()
    def logits(self, tokens, audio_features):
        return self.decoder.forward(tokens, audio_features)
