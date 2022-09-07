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
        start_dim = 1
        if len(input.shape) < 3:
            start_dim = 0

        x = th.flatten(input, start_dim=start_dim)
        output = self.nn(x)
        return output

class AE(nn.Module):
    def __init__(self) -> None:
        super(AE, self).__init__()

        self.input_shape = [512, 128]
        self.encoder = Encoder()

        decode_layers = [nn.Linear(1024, 4000, bias=False),
                    nn.ReLU(),
                    nn.Linear(4000, 8000, bias=False),
                    nn.ReLU(),
                    nn.Linear(8000, 16000, bias=False),
                    nn.ReLU(),
                    nn.Linear(16000, 32000, bias=False),
                    nn.ReLU(),
                    nn.Linear(32000, self.input_shape[0] * self.input_shape[1], bias=False),
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

        self.encoder = Encoder().to('cuda:0')
        for param in self.encoder.parameters():
            param.requires_grad = False

        # generate_layers = [nn.Linear(1024, 512),
        #             nn.ReLU(),
        #             nn.Linear(512, 512),
        #             nn.ReLU(),
        #             nn.Linear(512, 1024, bias=False),
        #             nn.ReLU(),
        #             nn.Linear(1024, 2048, bias=False),
        #             nn.ReLU(),
        #             nn.Linear(2048, 3072, bias=False)]

        # input = 1x1024
        generate_layers = [nn.Conv2d(1, 8, 3, padding='same', bias=False),
                        nn.BatchNorm1d(8),
                        nn.ReLU(),
                        nn.Conv2d(8, 16, 3, padding='same', bias=False),
                        nn.BatchNorm1d(16),
                        nn.ReLU(),
                        # upsampling block
                        nn.Conv2d(16, 64, 3, padding='same', bias=False),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv2d(64, 128, 3, padding='same', bias=False),
                        nn.ReLU(),
                        nn.Conv2d(128, 256, 3, padding='same', bias=False),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, 3, padding='same', bias=False),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        # de-coding/downsampling block
                        nn.Conv2d(256, 256, 3, padding='same', bias=False),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        nn.Conv2d(256, 512, 3, padding='same', bias=False),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        nn.Conv2d(512, 512, 3, padding='same', stride=2, bias=False),
                        nn.ReLU(),
                        nn.Conv2d(512, 1024, 3, padding='same', bias=False),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, 3, padding='same', bias=False),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 2048, 3, padding='same', stride=2, bias=False),
                        nn.ReLU(),
                        nn.Conv2d(2048, 3072, 3, padding='same', bias=False)]

        self.generator = nn.Sequential(*generate_layers).to('cuda:1')

    def forward(self, input):
        encoding = self.encoder(input)
        print(encoding.shape)
        output = self.generator(encoding.view(encoding.shape[0], 32, 32).to('cuda:1'))
        if len(input.shape) < 3:
            output = output.view(3072, 1024)
        else:
            output = output.view(input.shape[0], 3072, 1024)
        return output

    def load_encoder_weights(self, filename):
        self.encoder.load_state_dict(th.load(filename))
