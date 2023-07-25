import os
import re
import torch

import torch.nn as nn
import torch.nn.functional as F

from util.config import Config
from collections import OrderedDict
from kilonerf import KiloNeRF

class KiloNeRFSimple(nn.Module):
    def __init__(self, D=8, W=256, n_in=None, n_out=4, skips=[4], use_viewdirs=False, net_idx=None, config=None):
        """
        extend nerf to use 2 shared MLPs for processing points and view directions
        """

        super(KiloNeRFSimple, self).__init__()
        self.net_idx = net_idx
        self.D = D
        self.W = W
        self.skips = skips

        if 'auto' in skips[0]:
            self.skips = [4]
        else:
            self.skips = [int(x) for x in self.skips]

        self.name = f"NeRF{self.net_idx}({W}x{D}{self.skips})"
        self.input_ch = 3
        self.input_ch_views = 3
        self.output_ch = n_out

        if config.posEnc and config.posEnc[net_idx] and "RayMarch" in config.inFeatures[net_idx]:
            if config.posEnc[net_idx] == "nerf":
                freq = config.posEncArgs[net_idx].split("-")
                self.input_ch = (int(freq[0])) * 6 + 3
                self.input_ch_views = (int(freq[1])) * 6 + 3

        self.use_viewdirs = use_viewdirs

        # First shared MLP for processing points
        self.pts_linears_shared1 = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D // 2 - 1)]
        )

        # Second shared MLP for processing points
        self.pts_linears_shared2 = nn.ModuleList(
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D // 2 - 1, D - 1)]
        )

        # First shared MLP for processing view directions
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_views + W, W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, self.output_ch)

        for i, l in enumerate(self.pts_linears_shared1):
            nn.init.kaiming_normal_(l.weight)
        
        for i, l in enumerate(self.pts_linears_shared2):
            nn.init.kaiming_normal_(l.weight)

        for i, l in enumerate(self.views_linears):
            nn.init.kaiming_normal_(l.weight)

        # self.init_weights()

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears_shared1):
            h = self.pts_linears_shared1[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        for i, l in enumerate(self.pts_linears_shared2):
            h = self.pts_linears_shared2[i](h)
            h = F.relu(h)
            if i + len(self.pts_linears_shared1) in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def save_weights(self, path, name_suffix="", optimizer=None):
        torch.save(self.state_dict(), f"{path}{self.name}_{name_suffix}.weights")
        if optimizer is not None:
            torch.save(optimizer.state_dict(), f"{path}{self.name}_{name_suffix}.optimizer")

    def delete_saved_weights(self, path, optimizer=None):
        ckpts = [os.path.join(path, f) for f in sorted(os.listdir(os.path.join(path))) if
                 '.weights' in f and self.name in f and '_opt.weights' not in f]
        # keep the last 10 files just in case something happened during training
        for file in ckpts[:-10]:
            # and also keep the weights every 50k iterations
            epoch = int(file.split('.weights')[0].split('_')[-1])
            if epoch % 50000 == 0 and epoch > 0:
                continue
            os.remove(file)
            if optimizer is not None:
                os.remove(f"{file.split('.weights')[0]}.optimizer")

    def load_weights(self, path, device):
        print('Reloading Checkpoint from', path)
        model = torch.load(path, map_location=device)
        # no idea why, but sometimes torch.load returns an ordered_dict...
        if type(model) == type(OrderedDict()):
            self.load_state_dict(model)
        else:
            self.load_state_dict(model.state_dict())

    def load_optimizer(self, path, optimizer, device):
        if os.path.exists(path):
            print(f"Reloading optimizer checkpoint from {path}")
            optimizer_state = torch.load(path, map_location=device)
            optimizer.load_state_dict(optimizer_state)

    def load_specific_weights(self, path, checkpoint_name, optimizer=None, device=0):
        ckpts = [os.path.join(path, f) for f in sorted(os.listdir(os.path.join(path))) if
                 checkpoint_name in f and self.name in f]
        if len(ckpts) == 0:
            print("no Checkpoints found")
            return 0

        ckpt_path = ckpts[-1]

        self.load_weights(ckpt_path, device)

        if optimizer is not None:
            optim_path = f"{ckpt_path.split('.weights')[0]}.optimizer"
            self.load_optimizer(optim_path, optimizer, device)
        return 1

    def load_latest_weights(self, path, optimizer=None, device=0, config=None):
        ckpts = [os.path.join(path, f) for f in sorted(os.listdir(os.path.join(path))) if
                 '.weights' in f and self.name in f and not '_opt.weights' in f]
        if len(ckpts) == 0:
            print("no Checkpoints found")
            if config and config.preTrained and len(config.preTrained) > self.net_idx and config.preTrained[
                self.net_idx].lower() != "none":
                wpath = os.path.join(config.preTrained[self.net_idx], f"{self.name}.weights")
                if os.path.exists(wpath):
                    print("loading pretrained weights")
                    self.load_weights(wpath, device)
                else:
                    print(f"WARNING pretrained weights not found: {wpath}")
                opath = wpath = os.path.join(config.preTrained[self.net_idx], f"{self.name}.optimizer")
                if optimizer is not None and os.path.exists(opath):
                    self.load_optimizer(opath, optimizer, device)
            return 0
        ckpt_path = ckpts[-1]
        epoch = int(ckpt_path.split('.weights')[0].split('_')[-1])

        self.load_weights(ckpt_path, device)

        if optimizer is not None:
            optim_path = f"{ckpt_path.split('.weights')[0]}.optimizer"
            self.load_optimizer(optim_path, optimizer, device)

        return epoch