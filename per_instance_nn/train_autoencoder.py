import torch as th
import torch.nn as nn

from pi_features_model import AE

device = 'cuda' if th.cuda.is_available() else 'cpu'
autoencoder = AE()
autoencoder.to(device)
input_encoding = th.load("../brownie_variations_true_output.pt")
input_encoding.to(device)

EPOCHS = 10
learning_rate = 1e-3
optimizer = th.optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=1e-6)
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    
    output = autoencoder(input_encoding)
    loss = criterion(output, input_encoding)
    loss.backward()
    optimizer.step()

    print(f'Epoch: {epoch} - loss: {loss.item():.3f}')

autoencoder.save_encoder_weights("pi_encoder_weights.pt")
print('Training complete')
