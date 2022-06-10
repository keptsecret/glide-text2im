import torch as th

device = 'cuda' if th.cuda.is_available() else 'cpu'

delta_x : th.Tensor = th.load("brownie_pca_1e-1_s0.pt")
true_y : th.Tensor = th.load("brownie_true_output.pt")
perturb_y : th.Tensor = th.load("brownie_pca_outputs.pt")
delta_y : th.Tensor = perturb_y - true_y

A = th.rand([3*64*64, 512*128], requires_grad=True, device=device)
criterion = th.nn.MSELoss()
optimizer = th.optim.Adam([A], lr=1e-2)

EPOCHS = 10
for epoch in range(EPOCHS):
    running_loss = 0
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

    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 128:.5f}')
