import torch as th
import torch.nn as nn

from pi_features_model import AE

#device = 'cuda' if th.cuda.is_available() else 'cpu'
device = 'cpu'
dtype = th.float32

autoencoder = AE()
autoencoder = autoencoder.to(device, dtype=dtype)

EPOCHS = 40
learning_rate = 1e-4
optimizer = th.optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=1e-6)
scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, [20], gamma=0.1) # doesn't work
criterion = nn.MSELoss()

th.autograd.set_detect_anomaly(True)

for epoch in range(EPOCHS):
    running_loss = 0.0

    for i in range(181):
        optimizer.zero_grad()

        if i == 180:
            input_encoding = th.load("../brownie_true_input_[xf_out].pt")
        else:
            input_encoding = th.load("../brownie_variations/brownie_variation{i}_pca.pt")
        input_encoding = input_encoding[0]  # remove classifier-free guidance stack
        input_encoding = input_encoding.to(device, dtype=dtype)

        encoding = input_encoding - input_encoding.min(1, keepdim=True)[0]
        encoding /= encoding.max(1, keepdim=True)[0]
        encoding = encoding.to(device)
        output = autoencoder(encoding)
        #print(encoding)
        #print(output)

        loss = criterion(output.T, encoding.T)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
    scheduler.step()

    print(f'Epoch: {epoch+1} - loss: {running_loss / 181:.5f}')

autoencoder.save_encoder_weights("pi_encoder_weights.pt")
print('Training complete')
