"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
  
class HeadAttention(nn.Module):
    def __init__(self, n_embd, n_embd_head, attn_pdrop=0.1):
        super().__init__()
        self.q_lin = nn.Linear(n_embd, n_embd_head)
        self.k_lin = nn.Linear(n_embd, n_embd_head)
        self.v_lin = nn.Linear(n_embd, n_embd_head)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)

    def forward(self, x, padding_mask): 
        q = self.q_lin(x) # (Batch, Tokens, n_embd) -> (Batch, Tokens, n_embd_head)
        k = self.k_lin(x)
        v = self.v_lin(x)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        expanded_mask = padding_mask.unsqueeze(1)
        att = att.masked_fill(expanded_mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        return att @ v # (B, T, T) @ (B, T, n_embd_head) -> (B, T, n_embd_head)

#class MultiHeadAttentionSimple(nn.Module):
#    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1):
#        super().__init__()
#        assert n_embd % n_head == 0
#        # key, query, value projections for all heads, but in a batch
#        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
#        # output projection
#        self.c_proj = nn.Linear(n_embd, n_embd)
#        # regularization
#        self.attn_dropout = nn.Dropout(attn_pdrop)
#        self.resid_dropout = nn.Dropout(resid_pdrop)
#        self.n_head = n_head
#        self.n_embd = n_embd
#        
#    def forward(self, x, padding_mask=None):
#        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
#
#        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
#        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
#        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
#        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
#        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
#
#        #Done by me
#        mask = padding_mask.unsqueeze(1).unsqueeze(-1)
#        #mask = padding_mask.view(1,1,T,T)
#
#        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
#        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#        att = att.masked_fill(mask[:,:,:T,:T] == 1, float('-inf'))
#        att = F.softmax(att, dim=-1)
#        att = self.attn_dropout(att)
#        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
#        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
#
#        # output projection
#        y = self.resid_dropout(self.c_proj(y))
#        return y

class MultiHeadAttentionSimple(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads as a list
        self.heads = nn.ModuleList([HeadAttention(n_embd, n_embd // n_head, attn_pdrop) for _ in range(n_head)])
        self.c_proj = nn.Linear(n_embd, n_embd)  # output projection to integrate head outputs
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def forward(self, x, padding_mask):
        y = torch.cat([h(x, padding_mask) for h in self.heads], dim=-1)  # (B, T, n_embd)
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttentionSimple(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.ln_2 = nn.LayerNorm(n_embd)
        #Feed-Forward
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),  # ffw hidden layer size is fixed to 4*n_embd
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = nn.GELU(),
            dropout = nn.Dropout(resid_pdrop),
        ))
        
    def forward(self, x, padding_block):
        x = x + self.attn(self.ln_1(x), padding_block)
        m = self.mlp  # just a shorter name
        x = x +  m.dropout(m.c_proj(m.act(m.c_fc(self.ln_2(x)))))
        return x

class Net(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.pad_ind = params.pad_ind
        self.transformer = nn.ModuleDict(dict(
            #Target embedding
            wte = nn.Embedding(params.vocab_size, params.embedding_dim),
            #Position embedding
            wpe = nn.Embedding(200, params.embedding_dim),
            drop = nn.Dropout(0.1),
            h = nn.ModuleList([Block(params.embedding_dim, params.num_heads, 0.1, 0.1) for _ in range(params.num_layers)]),
            ln_f = nn.LayerNorm(params.embedding_dim),
        ))
        self.lm_head = nn.Linear(params.embedding_dim, params.number_of_tags, bias=False)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

    def forward(self, inputs, outputs=None):
        padding_mask = inputs == self.pad_ind
        device = inputs.device
        #Batch size and token size
        b, t = inputs.size()
        #Create positional tensor with 0-t values
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        #Input with embeddings
        tok_emb = self.transformer.wte(inputs) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)
        #Dropout of the token+pos output
        x = self.transformer.drop(tok_emb + pos_emb)
        #Multi-Head Attention
        for block in self.transformer.h:
            x = block(x, padding_mask)
        #Layer norm
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return F.log_softmax(logits.view(logits.size(dim=0)*logits.size(dim=1), -1), dim=-1)

def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs from the model and labels for all tokens. Exclude loss terms
    for PADding tokens.

    Args:
        outputs: (Variable) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (Variable) dimension batch_size x seq_len where each element is either a label in [0, 1, ... num_tag-1],
                or -1 in case it is a PADding token.

    Returns:
        loss: (Variable) cross entropy loss for all tokens in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.view(-1)

    # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0).float()

    # indexing with negative values is not supported. Since PADded tokens have label -1, we convert them to a positive
    # number. This does not affect training, since we ignore the PADded tokens with the mask.
    labels = labels % outputs.shape[1]

    num_tokens = int(torch.sum(mask))

    # compute cross entropy loss for all tokens (except PADding tokens), by multiplying with mask.
    return -torch.sum(outputs[range(outputs.shape[0]), labels]*mask)/num_tokens


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all tokens. Exclude PADding terms.

    Args:
        outputs: (np.ndarray) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (np.ndarray) dimension batch_size x seq_len where each element is either a label in
                [0, 1, ... num_tag-1], or -1 in case it is a PADding token.

    Returns: (float) accuracy in [0,1]
    """

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.ravel()

    # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0)

    # np.argmax gives us the class predicted for each token by the model
    outputs = np.argmax(outputs, axis=1)
    print("OUTPUTS")
    print(outputs)
    print("LABELS")
    print(labels)

    # compare outputs with labels and divide by number of tokens (excluding PADding tokens)
    return np.sum(outputs == labels)/float(np.sum(mask))


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}