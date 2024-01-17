import torch as T
import torch.nn as nn
import numpy as np


class AttentionEncoder(nn.Module):
    def __init__(self, device, input_dim: int,
                 hidden_size: int=64,
                 num_lstm_layers: int=1,
                 denoising: bool=False,
                 num_features: int=11,
                 num_lags: int=10,
                 matrix_rep: bool=True,
                 architecture: str="lstm"):
        super(AttentionEncoder, self).__init__()
        self.hidden_dim = hidden_size
        self.layer_dim = num_lstm_layers
        self.add_noise = denoising
        self.num_features = num_features
        self.num_lags = num_lags
        self.matrix_rep = matrix_rep
        self.architecture = architecture
        if not self.matrix_rep:
            self.input_dim = input_dim // num_lags
        else:
            self.input_dim = input_dim

        if architecture == "lstm":
            self.rnn = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_size, num_layers=1)
        else:
            self.rnn = nn.GRU(input_size=self.input_dim, hidden_size=hidden_size, num_layers=1)
        self.attention = nn.Linear(
            in_features=2 * hidden_size + num_lags,
            out_features=1
        )
        self.softmax = nn.Softmax(dim=1)
        self.to(device)


    def _get_noise(self, x, sigma=0.01, p=0.1):
        normal = sigma * T.randn(x.shape)
        mask = np.random.uniform(size=x.shape)
        mask = (mask < p).astype(int)
        noise = normal * T.tensor(mask)
        return noise

    def forward(self, x, exogenous=None, device="cpu"):
        if len(x.shape) > 2:
            x = x.reshape(x.size(0), x.size(1), x.size(2))
        else:
            x = x.reshape(x.size(0), x.size(1) // self.num_features, x.size(1) // self.num_lags)

        ht = nn.init.xavier_normal_(T.zeros(1, x.size(0), self.hidden_dim)).to(device)
        ct = nn.init.xavier_normal_(T.zeros(1, x.size(0), self.hidden_dim)).to(device)

        attention = T.zeros(x.size(0), self.num_lags, self.input_dim).to(device)
        input_encoded = T.zeros(x.size(0), self.num_lags, self.hidden_dim).to(device)

        if self.add_noise and self.training:
            x += self._get_noise(x).to(device)

        for t in range(x.size(1)):
            x_ = T.cat(
                (ht.repeat(self.input_dim, 1, 1).permute(1, 0, 2),
                 ct.repeat(self.input_dim, 1, 1).permute(1, 0, 2),
                 x.permute(0, 2, 1).to(device)), dim=2).to(device)  # bs * input_size * (2 * hidden_dim + seq_len)

            et = self.attention(x_.view(-1, self.hidden_dim * 2 + self.num_lags))  # bs * input_size * 1
            at = self.softmax(et.view(-1, self.input_dim)).to(device)  # (bs * input_size)

            weighted_input = T.mul(at, x[:, t, :].to(device))  # (bs * input_size)

            self.rnn.flatten_parameters()
            if self.architecture == "lstm":
                _, (ht, ct) = self.rnn(weighted_input.unsqueeze(0), (ht, ct))
            else:
                _, ht = self.rnn(weighted_input.unsqueeze(0), ht)

            input_encoded[:, t, :] = ht
            attention[:, t, :] = at

        return attention, input_encoded

class AttentionDecoder(nn.Module):
    def __init__(self,
                 device,
                 encoder_hidden_size: int = 64,
                 decoder_hidden_size: int = 64,
                 num_lags: int = 10,
                 out_dim: int = 5,
                 architecture="lstm"
                 ):
        super(AttentionDecoder, self).__init__()

        self.encoder_dim = encoder_hidden_size
        self.decoder_dim = decoder_hidden_size
        self.num_lags = num_lags
        self.architecture = architecture

        self.attention = nn.Sequential(
            nn.Linear(2 * decoder_hidden_size + encoder_hidden_size, encoder_hidden_size),
            nn.Tanh(),
            nn.Linear(encoder_hidden_size, 1)
        )
        if architecture == "lstm":
            self.rnn = nn.LSTM(input_size=out_dim, hidden_size=decoder_hidden_size)
        else:
            self.rnn = nn.GRU(input_size=out_dim, hidden_size=decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + out_dim, out_dim)
        self.fc_out = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_dim)
        self.softmax = nn.Softmax(dim=1)

        self.fc.weight.data.normal_()
        self.to(device)

    def forward(self, device, x_encoded, exogenous=None, y_history=None):
        ht = nn.init.xavier_normal_(T.zeros(1, x_encoded.size(0), self.encoder_dim)).to(device)
        ct = nn.init.xavier_normal_(T.zeros(1, x_encoded.size(0), self.encoder_dim)).to(device)

        context = T.autograd.Variable(T.zeros(x_encoded.size(0), self.encoder_dim))

        for t in range(self.num_lags):
            x = T.cat((
                ht.repeat(self.num_lags, 1, 1).permute(1, 0, 2),
                ct.repeat(self.num_lags, 1, 1).permute(1, 0, 2),
                x_encoded.to(device)), dim=2)

            x = self.softmax(
                self.attention(
                    x.view(-1, 2 * self.decoder_dim + self.encoder_dim)
                ).view(-1, self.num_lags))

            context = T.bmm(x.unsqueeze(1), x_encoded.to(device))[:, 0, :]  # bs * encoder_dim

            y_tilde = self.fc(T.cat((context.to(device), y_history[:, t].to(device)),
                                        dim=1))  # bs * out_dim

            self.rnn.flatten_parameters()
            if self.architecture == "lstm":
                _, (ht, ct) = self.rnn(y_tilde.unsqueeze(0), (ht, ct))
            else:
                _, ht = self.rnn(y_tilde.unsqueeze(0), ht)

        out = self.fc_out(T.cat((ht[0], context.to(device)), dim=1))  # seq + 1

        return out


class DualAttentionAutoEncoder(nn.Module):
    def __init__(self, device, input_dim: int, architecture: str="lstm", matrix_rep: bool=True):
        super(DualAttentionAutoEncoder, self).__init__()
        self.encoder = AttentionEncoder(device=device, input_dim=input_dim, architecture=architecture, matrix_rep=matrix_rep)
        self.decoder = AttentionDecoder(device=device, architecture=architecture)
        self.device = device
        self.to(device)

    def forward(self, x, exogenous=None, device="cpu", y_hist=None):
        attentions, encoder_out = self.encoder(x, exogenous, device)
        outputs = self.decoder(device, encoder_out, exogenous, y_hist)
        return outputs