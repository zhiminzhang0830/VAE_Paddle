import random

import numpy as np
import paddle
from PIL import Image

import paddle.nn.functional as F


def set_random_seed(seed: int):
    """Set numpy, random, paddle random_seed to given seed.

    Args:
        seed (int): Random seed.
    """
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class Encoder(paddle.nn.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        self.FC_mean = paddle.nn.Linear(in_features=128, out_features=128)
        self.FC_var = paddle.nn.Linear(in_features=128, out_features=128)
        self.LeakyReLU = paddle.nn.LeakyReLU(negative_slope=0.2)
        self.training = True

        modules = []
        modules.append(
            paddle.nn.Sequential(
                paddle.nn.Conv2D(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                paddle.nn.LeakyReLU(negative_slope=0.2),
                paddle.nn.Conv2D(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                ),
                paddle.nn.LeakyReLU(negative_slope=0.2),
                paddle.nn.Conv2D(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=7,
                    stride=1,
                    padding=0,
                ),
                paddle.nn.LeakyReLU(negative_slope=0.2),
            )
        )
        self.modules = paddle.nn.Sequential(*modules)



    def forward(self, x):
        x = x.reshape([x.shape[0], 1, 28, 28])
        x = self.modules(x)
        x = paddle.flatten(x=x, start_axis=1)
        h_ = x
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        return mean, log_var


class Decoder(paddle.nn.Layer):
    def __init__(self):
        super(Decoder, self).__init__()
        self.FC_hidden = paddle.nn.Linear(
            in_features=128, out_features=256
        )
        self.FC_hidden2 = paddle.nn.Linear(
            in_features=256, out_features=512
        )
        self.FC_output = paddle.nn.Linear(
            in_features=512, out_features=784
        )
        self.LeakyReLU = paddle.nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):

        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))
        x_hat = paddle.nn.functional.sigmoid(x=self.FC_output(h))
        return x_hat


class Model(paddle.nn.Layer):
    def __init__(self):
        super(Model, self).__init__()
        self.Encoder = Encoder()
        self.Decoder = Decoder()

    def reparameterization(self, mean, var):
        epsilon = paddle.randn(shape=var.shape, dtype=var.dtype)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, paddle.exp(x=0.5 * log_var))
        x_hat = self.Decoder(z)
        return x_hat, mean, log_var


    def sample(self, num_samples: int):
        z = paddle.randn([num_samples, 128])
        samples = self.Decoder(z)
        samples = samples.reshape([num_samples, 1, 28, 28])
        return samples



def loss_function(x, x_hat, mean, log_var, kld_weight):
    reproduction_loss = F.binary_cross_entropy(input=x_hat, label=x, reduction="sum")
    kld_loss = -0.5 * paddle.sum(x=1 + log_var - mean.pow(y=2) - log_var.exp())
    loss = reproduction_loss + kld_weight * kld_loss
    return loss, reproduction_loss.detach(), kld_loss.detach()


if __name__ == "__main__":
    set_random_seed(42)

    n_epoch = 50
    kld_weight = 1.0
    n_sample = 10
    batch_size = 100

    model = Model()
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=0.001
    )

    dataset = paddle.vision.datasets.MNIST(
        mode="train",
        download=True,
        transform=paddle.vision.transforms.Compose(
            [paddle.vision.transforms.ToTensor(), ] #, paddle.vision.transforms.Pad(2)]
        ),
    )

    train_loader = paddle.io.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    model.train()
    for epoch in range(n_epoch):
        total_loss = 0.0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view([x.shape[0], -1])
            x_hat, mean, log_var = model(x)
            loss, recon_loss, kld_loss = loss_function(x, x_hat, mean, log_var, kld_weight)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if batch_idx % 100 == 0 or batch_idx + 1 == len(train_loader):
                print(
                    f"Epoch: [{epoch}/{n_epoch}] "
                    f"Step: [{batch_idx+1}/{len(train_loader)}]"
                    f"Loss: {loss.item():.4f} "
                    f"Reconstruction_Loss: {recon_loss.item():.4f} "
                    f"KLD_Loss: {kld_loss.item():.4f}"
                )
            total_loss += loss.item()
        print(f"Epoch: [{epoch}/{n_epoch}] average loss: {total_loss / len(train_loader) / batch_size}")

    model.eval()
    with paddle.no_grad():
        generated_images = model.sample(n_sample)
    generated_images = generated_images.numpy() * 255
    generated_images = generated_images.astype(np.uint8)

    for idx in range(n_sample):
        image = Image.fromarray(generated_images[idx][0])
        image.save(f"./generated_image_{idx}.png")
