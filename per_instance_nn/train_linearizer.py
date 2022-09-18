import torch as th
import torch.nn as nn

from pi_features_model import PerInstanceLinearizerV2 as PerInstanceLinearizer

# device = 'cuda' if th.cuda.is_available() else 'cpu'
# device = 'cpu'
# device = 'cuda:0'
dtype = th.float32

model = PerInstanceLinearizer()
model.load_encoder_weights("pi_betterencoder_weights.pt")
model = model.to(dtype=dtype)

EPOCHS = 40
learning_rate = 2e-4
optimizer = th.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
scheduler = th.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.2)

th.autograd.set_detect_anomaly(True)

x = th.load("../brownie_true_input_[xf_out].pt").to(dtype=dtype)
x = x[0]  # remove classifier-free guidance stack

y = th.load("../brownie_variations_true_output.pt").to("cuda:1", dtype=dtype)
y = y.flatten()

def criterion(A, dxs, dys):
    loss_fn = nn.MSELoss()

    value = 0.0
    for dx, dy in zip(dxs, dys):
        enc_dx = model.encode(dx) #.to('cuda:0'))
        pred_dy = th.matmul(A, enc_dx.view(1024)) #.to('cpu'))
        value += loss_fn(pred_dy, dy)

    return value / dxs.shape[0]

for epoch in range(EPOCHS):
    running_loss = 0.0

    for i in range(180):
        optimizer.zero_grad()

        dx = th.load(f"../brownie_variations/brownie_variation{i}_pca.pt").to('cuda:0', dtype=dtype)
        dx = dx.T[:, None, :].repeat(1, 512, 1)

        A_mat = model(x)

        new_y = th.load(f"../brownie_variations_outputs/brownie_variation{i}_pca_output.pt").to('cuda:1', dtype=dtype)
        dy = new_y - y

        loss = criterion(A_mat.to('cuda:0'), dx.to('cuda:0'), dy.to('cuda:0'))
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
        print(f"\rIter: {i} - loss: {loss.item():.5f}", end='')
    scheduler.step()

    print(f'\rEpoch: {epoch+1} - loss: {running_loss / 180:.5f}')

th.save(model.state_dict(), "pi_linearizer_weights_v2.pt")
print('Training complete')
