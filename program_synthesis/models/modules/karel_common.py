import torch
import torch.nn as nn


class LGRLTaskEncoder(nn.Module):
    '''Implements the encoder from:

    Leveraging Grammar and Reinforcement Learning for Neural Program Synthesis
    https://openreview.net/forum?id=H1Xw62kRZ
    '''

    def __init__(self, args):
        super(LGRLTaskEncoder, self).__init__()

        self.input_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=15, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(), )
        self.output_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=15, out_channels=32, kernel_size=3, padding=1),
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

        self.fc = nn.Linear(64 * 18 * 18, 512)

    def forward(self, input_grid, output_grid):
        batch_dims = input_grid.shape[:-3]
        input_grid = input_grid.contiguous().view(-1, 15, 18, 18)
        output_grid  = output_grid.contiguous().view(-1, 15, 18, 18)

        input_enc = self.input_encoder(input_grid)
        output_enc = self.output_encoder(output_grid)
        enc = torch.cat([input_enc, output_enc], 1)
        enc = enc + self.block_1(enc)
        enc = enc + self.block_2(enc)

        enc = self.fc(enc.view(*(batch_dims + (-1,))))
        return enc


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


def compress_trace(grids):
    result = []
    last_indices = set()
    for grid in grids:
        grid = [int(x) for x in grid]
        indices = set(grid)
        added = indices - last_indices
        removed = last_indices - indices
        if len(added) + len(removed) < len(indices):
            result.append({'plus': sorted(added), 'minus': sorted(removed)})
        else:
            result.append(grid)
        last_indices = indices
    return result
