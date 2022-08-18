"""
Trains a local linearization matrix based on meaningful perturbations of the encoding input
"""

import torch as th

device = 'cuda' if th.cuda.is_available() else 'cpu'

print("Setting up tensors")
delta_x : th.Tensor = th.load("airplane_pca.pt")
true_y : th.Tensor = th.load("airplane_true_output.pt")
perturb_y : th.Tensor = th.load("airplane_pca_outputs.pt")
delta_y : th.Tensor = perturb_y - th.flatten(true_y)

A = th.rand([3*32*32, 512*128], requires_grad=True, device=device)
criterion = th.nn.MSELoss()
optimizer = th.optim.Adam([A], lr=1e-2)
scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

print("Starting fitting")
EPOCHS = 40
for epoch in range(EPOCHS):
    running_loss = 0
    max = 0
    min = 1000000
    for i in range(delta_x.shape[0]):
        x = delta_x[i]
        x = x[None, :].repeat(512, 1).flatten().to(device=device)
        x = x[:, None]

        optimizer.zero_grad()

        y_real = delta_y[i]
        y_real = y_real[None, :]
        y_pred = th.matmul(A, x).transpose(0, 1)

        loss = criterion(y_pred, y_real)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if loss.item() > max:
            max = loss.item()
        if loss.item() < min:
            min = loss.item()

    print(f'[{epoch + 1}, {i + 1:5d}]\tloss: {running_loss / 128:.5f}\tmin: {min:.5f}\tmax: {max:.5f}')
    scheduler.step()

print("Finished fitting, saving matrix...")
th.save(A, "airplane_A.pt")
