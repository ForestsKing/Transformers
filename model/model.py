from model.decoder import Decoder
from model.encoder import Encoder
from torch import nn


class Model(nn.Module):
    def __init__(self, attention='FullAttention', d_k=64, d_v=64, d_model=512, d_ff=2048, n_heads=8,
                 en_layer=3, de_layer=2, d_feature=7, d_mark=4, dropout=0.1):
        super(Model, self).__init__()

        self.encoder = Encoder(attention=attention, d_k=d_k, d_v=d_v, d_model=d_model, d_ff=d_ff, n_heads=n_heads,
                               n_layer=en_layer, d_feature=d_feature, d_mark=d_mark, dropout=dropout)
        self.decoder = Decoder(attention=attention, d_k=d_k, d_v=d_v, d_model=d_model, d_ff=d_ff, n_heads=n_heads,
                               n_layer=de_layer, d_feature=d_feature, d_mark=d_mark, dropout=dropout)

        self.projection = nn.Linear(d_model, d_feature, bias=True)

    def forward(self, enc_x, enc_mark, dec_in, dec_mark):
        enc_outputs = self.encoder(enc_x, enc_mark)
        dec_outputs = self.decoder(dec_in, dec_mark, enc_outputs)
        dec_outputs = self.projection(dec_outputs)
        return dec_outputs
