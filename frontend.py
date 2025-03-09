import streamlit as st
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

ngpu = 0
nc = 3 # Number of channels in the training images. For color images this is 3
nz = 100 # Size of z latent vector (i.e. size of generator input)
ngf = 64 # Size of feature maps in generator
device = "cpu"

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)

netG = Generator(ngpu).to(device)

netG.load_state_dict(torch.load("models/generator_500_epochs.pth"))
netG.eval()

fixed_noise = torch.normal(mean=0.2, std=1, size=(64, nz, 1, 1))

def generate_images():
    """Generate fake images and return as a numpy array."""
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
        img_grid = vutils.make_grid(fake, padding=2, normalize=True)
    return np.transpose(img_grid.numpy(), (1, 2, 0))

st.title("GAN Fake Image Generator")

if st.button("Reload Images"):
    st.session_state["img"] = generate_images()

if "img" not in st.session_state:
    st.session_state["img"] = generate_images()

fig, ax = plt.subplots()
ax.axis("off")
ax.set_title("Fake Images")
ax.imshow(st.session_state["img"])
st.pyplot(fig)