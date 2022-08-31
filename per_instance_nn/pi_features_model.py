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
                    nn.Linear(32000, self.input_shape[0] * self.input_shape[1])]

        self.decoder = nn.Sequential(*decode_layers)

    def forward(self, input):
        x = self.encoder(input)
        output = self.decoder(x)
        return output

    def save_encoder_weights(self, filename):
        th.save(self.encoder.state_dict(), filename)
