import torch as th
from torchvision import transforms

def show_images(batch: th.Tensor, filename):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    # reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    # display(Image.fromarray(reshaped.numpy()))
    im = transforms.ToPILImage()(scaled)
    im.save(filename)

device = 'cuda' if th.cuda.is_available() else 'cpu'
print(device)

# x = th.load("brownie_pca_1e-1_s0.pt").to(dtype=th.float16, device=device)
#x = th.load("brownie_in_s0.pt")
A = th.load("airplane_A.pt", map_location=th.device('cpu'))
U, S, Vh = th.linalg.svd(A, full_matrices=False)
print(U.shape)
print(S)
print(Vh.shape)
new_S = th.tensor([x if x > 60 else 0 for x in S])
th.save(Vh, 'airplane_Vh.pt')
print(new_S)
d = th.dist(A, U @ th.diag(new_S) @ Vh)
print(d)
# true_y = th.load("brownie_true_output.pt")

# x = x[1]
# x = x[None, :].repeat(512, 1).flatten().to(dtype=th.float16, device=device)
# x = x[:, None]
# y = th.matmul(A, x).view(size=[3, 32, 32])
# combined_y = true_y + y

# show_images(combined_y, "test_image2.png")
