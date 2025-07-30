import torch

class Time2Vec(nn.Module):
    """
    Time2Vec: map scalar time t -> d_model-dimensional vector.
    v0 = w0 * t + b0  (linear)
    v[i] = sin(w[i] * t + b[i])  for i=1..d_model-1
    """
    def __init__(self, d_model):
        super().__init__()
        # one linear + (d_model-1) periodic features
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w  = nn.Parameter(torch.randn(d_model-1))
        self.b  = nn.Parameter(torch.zeros(d_model-1))

    def forward(self, t):
        """
        t: (B, L)  - scalar "time since first detection" per event
        returns (B, L, d_model)
        """
        # linear term
        v0 = self.w0 * t + self.b0                          # (B, L)
        # periodic terms
        vp = torch.sin(t.unsqueeze(-1) * self.w + self.b)   # (B, L, d_model-1)
        return torch.cat([v0.unsqueeze(-1), vp], dim=-1)    # (B, L, d_model)


class BaselineCLS(nn.Module):
    """
    Paper: arxiv.org/abs/2507.16088v2
    """

    def __init__(self, d_model, n_heads, n_layers,
                 num_classes, dropout, max_len=None, mode=None):
        super().__init__()
        self.in_proj  = nn.Linear(7, d_model)
        self.cls_tok  = nn.Parameter(torch.zeros(1,1,d_model))
        
        # replace SinCos PE with Time2Vec on the dt channel
        self.time2vec = Time2Vec(d_model).to(torch.device("cuda"))
        
        enc_layer      = nn.TransformerEncoderLayer(
                            d_model, n_heads, d_model*4,
                            dropout, batch_first=True)
        self.encoder  = nn.TransformerEncoder(enc_layer, n_layers)
        self.norm     = nn.LayerNorm(d_model)
        self.head     = nn.Linear(d_model, num_classes)
        
        
        self.classification = True if mode == 'photo' else False
        if self.classification:
            self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, pad_mask):
        """
        x: (B, L, 7)  - the raw event tensor from build_event_tensor
            channels: [ dt, dt_prev, logf, logfe, one-hot-band(3) ]
        pad_mask: (B, L) boolean
        """
        B,L,_ = x.shape

        # project into model dim
        h = self.in_proj(x)                     # (B, L, d_model)
        # extract the *continuous* log1p dt feature
        t = x[..., 0]                           # (B, L)

        # compute the learned time embedding:
        te = self.time2vec(t)                  # (B, L, d_model)

        # add it:
        h = h + te                              # (B, L, d_model)

        # prepend a learned CLS token:
        tok = self.cls_tok.expand(B,-1,-1)      # (B,1,d_model)
        h   = torch.cat([tok, h], dim=1)        # (B, L+1, d_model)
        
        # adjust padding mask to account for CLS at idx=0
        pad = torch.cat(
            [torch.zeros(B,1, device=torch.device("cuda"), dtype=torch.bool),
             pad_mask], dim=1
        )

        # encode
        z = self.encoder(h, src_key_padding_mask=pad)  # (B, L+1, d_model)
        
        output = self.norm(z[:,0]) # (B, d_model )
        
        if self.classification:
            # classification from the CLS token
            output = self.fc(output) # (B, num_classes)     
            
        return output
                