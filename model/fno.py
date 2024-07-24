import torch
import torch.nn as nn

class FourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(FourierLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

    def forward(self, x):
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft(x, dim=-1, norm="ortho")
        out_ft = torch.zeros((batch_size, self.out_channels, x_ft.size(-1)), dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum('bix,iox->box', x_ft[:, :, :self.modes], self.weights)
        x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1, norm="ortho")
        return x

class FNO(nn.Module):
    def __init__(self, in_channels, out_channels, modes, width):
        super(FNO, self).__init__()
        self.conv1 = nn.Conv1d(1, width, 1)
        self.conv2 = nn.Conv1d(width, width, 1)
        self.conv3 = nn.Conv1d(width, width, 1)
        self.conv4 = nn.Conv1d(width, width, 1)
        self.fourier_layer = FourierLayer(width, width, modes)
        self.conv5 = nn.Conv1d(width, out_channels, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch_size, 1, 4260]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)   # [batch_size, width, 4260]
        x = self.fourier_layer(x)  # [batch_size, width, 4260]
        x = torch.relu(x)
        x = self.conv5(x)   # [batch_size, out_channels, 4260]
        return x