import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_Mnist(nn.Module):
    def __init__(self):
        super(VAE_Mnist, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 30)
        self.fc3 = nn.Linear(400, 30)

        # self.fc4 = nn.Linear(30, 400)
        # self.fc5 = nn.Linear(400, 784)
        self.decoder = Mnist_Decoder()

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # h = F.relu(self.fc4(z))
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var


class Mnist_Decoder(nn.Module):
    def __init__(self):
        super(Mnist_Decoder, self).__init__()
        self.fc4 = nn.Linear(30, 400)
        self.fc5 = nn.Linear(400, 784)

    def forward(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))
		
		
class VAE_xray(nn.Module):
    def __init__(self):
        super(VAE_xray, self).__init__()
        self.fc1 = nn.Linear(32*32, 500)
        self.fc2 = nn.Linear(500, 30)
        self.fc3 = nn.Linear(500, 30)

        # self.fc4 = nn.Linear(20, 400)
        # self.fc5 = nn.Linear(400, 784)
        self.decoder = xray_Decoder()

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # h = F.relu(self.fc4(z))
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var


class xray_Decoder(nn.Module):
    def __init__(self):
        super(xray_Decoder, self).__init__()
        self.fc4 = nn.Linear(30, 500)
        self.fc5 = nn.Linear(500, 32*32)

    def forward(self, z):
        h = F.relu(self.fc4(z))
        result = torch.sigmoid(self.fc5(h))
        return result
		
if __name__ == "__main__":
    model = VAE_xray()
    x = torch.rand(3, 32*32) # pytorch: [N, C, H, W]
    result = model(x)