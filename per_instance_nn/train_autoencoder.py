import torch as th
import torch.nn as nn

from pi_features_model import AE

#device = 'cuda' if th.cuda.is_available() else 'cpu'
device = 'cpu'
dtype = th.float32

autoencoder = AE()
autoencoder = autoencoder.to(device, dtype=dtype)
input_encoding = th.load("../brownie_true_input_[xf_out].pt")
input_encoding = input_encoding[0]  # remove classifier-free guidance stack
input_encoding = input_encoding.to(device, dtype=dtype)

EPOCHS = 40
learning_rate = 1e-4
optimizer = th.optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=1e-6)
scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, [20], gamma=0.1) # doesn't work
criterion = nn.MSELoss()

th.autograd.set_detect_anomaly(True)

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    encoding = input_encoding - input_encoding.min(1, keepdim=True)[0]
    encoding /= encoding.max(1, keepdim=True)[0]
    encoding = encoding.to(device)
    output = autoencoder(encoding)
    #print(encoding)
    #print(output)

    loss = criterion(output.T, encoding.T)
    loss.backward()

    optimizer.step()
    scheduler.step()

    print(f'Epoch: {epoch+1} - loss: {loss.item():.5f}')

autoencoder.save_encoder_weights("pi_encoder_weights.pt")
print('Training complete')
