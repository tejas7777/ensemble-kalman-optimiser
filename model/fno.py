# import torch
# import torch.nn as nn

# class FourierLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, modes):
#         super(FourierLayer, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes = modes
#         self.scale = (1 / (in_channels * out_channels))
#         self.weights = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

#     def forward(self, x):
#         batch_size = x.shape[0]
#         x_ft = torch.fft.rfft(x, dim=-1, norm="ortho")
#         out_ft = torch.zeros_like(x_ft, dtype=torch.cfloat)
#         out_ft[:, :, :self.modes] = x_ft[:, :, :self.modes] @ self.weights
#         x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1, norm="ortho")
#         return x

# class FNO(nn.Module):
#     def __init__(self, in_channels, out_channels, modes, width):
#         super(FNO, self).__init__()
#         self.fourier_layer = FourierLayer(in_channels, in_channels, modes)
#         self.conv1 = nn.Conv1d(in_channels, width, 1)  # Adjusted input channels
#         self.conv2 = nn.Conv1d(width, width, 1)
#         self.conv3 = nn.Conv1d(width, out_channels, 1)

#     def forward(self, x):
#         x = x.permute(0, 2, 1)  #batch_size, 4260] -> [batch_size, 4260, in_channels]
#         x = self.conv1(x)
#         x = self.fourier_layer(x)
#         x = torch.relu(x)
#         x = self.conv2(x)
#         x = torch.relu(x)
#         x = self.conv3(x)
#         x = x.permute(0, 2, 1)  # Change shape back to [batch_size, 4260, out_channels]
#         return x

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
        self.conv1 = nn.Conv1d(in_channels, width, 1)
        self.fourier_layer = FourierLayer(width, width, modes)
        self.conv2 = nn.Conv1d(width, out_channels, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Change input from [batch_size, 4260] to [batch_size, 1, 4260]
        x = self.conv1(x)
        x = self.fourier_layer(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = x.squeeze(1)  # Change shape back to [batch_size, 4260, out_channels]
        return x
