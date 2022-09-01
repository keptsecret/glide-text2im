from turtle import forward
import torch as th
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()

        self.input_shape = [512, 128]
        layers = [nn.Linear(self.input_shape[0] * self.input_shape[1], 16000),
                    nn.ReLU(),
                    nn.Linear(16000, 8000),
                    nn.ReLU(),
                    nn.Linear(8000, 4000),
                    nn.ReLU(),
                    nn.Linear(4000, 2000),
                    nn.ReLU(),
                    nn.Linear(2000, 1024),
                    nn.ReLU()]

        self.nn = nn.Sequential(*layers)

    def forward(self, input):
        x = th.flatten(input)
        output = self.nn(x)
        return output

class AE(nn.Module):
    def __init__(self) -> None:
        super(AE, self).__init__()

        self.input_shape = [512, 128]
        self.encoder = Encoder()

        decode_layers = [nn.Linear(1024, 4000),
                    nn.ReLU(),
                    nn.Linear(4000, 8000),
                    nn.ReLU(),
                    nn.Linear(8000, 16000),
                    nn.ReLU(),
                    nn.Linear(16000, 32000),
                    nn.ReLU(),
                    nn.Linear(32000, self.input_shape[0] * self.input_shape[1]),
                    nn.Sigmoid()]

        self.decoder = nn.Sequential(*decode_layers)

    def forward(self, input):
        x = self.encoder(input)
        x = self.decoder(x)
        output = x.view(512, 128)
        return output

    def save_encoder_weights(self, filename):
        th.save(self.encoder.state_dict(), filename)

class PerInstanceLinearizer(nn.Module):
    def __init__(self) -> None:
        super(PerInstanceLinearizer, self).__init__()

        self.encoder = Encoder()
        for param in self.encoder.parameters():
            param.requires_grad = False

        generate_layers = [nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 1024, bias=False),
                    nn.ReLU(),
                    nn.Linear(1024, 2048, bias=False),
                    nn.ReLU(),
                    nn.Linear(2048, 3072, bias=False)]

        self.generator = nn.Sequential(*generate_layers)

    def forward(self, input):
        encoding = self.encoder(input)
        x = self.generator(encoding)
        output = x.view(3, 32, 32)
        return output

    def load_encoder_weights(self, filename):
        self.encoder.load_state_dict(th.load(filename))
