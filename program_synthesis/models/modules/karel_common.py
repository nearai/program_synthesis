import torch.nn as nn

class PResNetGridEncoder(nn.Module):

    def __init__(self, args):
        super(PResNetGridEncoder, self).__init__()

        # TODO: deduplicate with one in karel.py
        # (which will break checkpoints?)
        self.initial_conv = nn.Conv2d(
            in_channels=15, out_channels=64, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, padding=1))
            for _ in range(3)
        ])
        self.grid_fc = nn.Linear(64 * 18 * 18, 256)

    def forward(self, grids):
        # grids: batch size x 15 x 18 x 18
        enc = self.initial_conv(grids)
        for block in self.blocks:
            enc = enc + block(enc)
        enc = self.grid_fc(enc.view(enc.shape[0], -1))
        return enc


class LGRLGridEncoder(nn.Module):
    def __init__(self, args):
        super(LGRLGridEncoder, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=15, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(), )

        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(), )
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(), )

        self.grid_fc = nn.Linear(64 * 18 * 18, 256)

    def forward(self, grids):
        # grids: batch size x 15 x 18 x 18
        enc = self.initial_conv(grids)
        enc = enc + self.block_1(enc)
        enc = enc + self.block_2(enc)
        enc = self.grid_fc(enc.view(enc.shape[0], -1))
        return enc


def none_fn(*args, **kwargs):
    return None
