import json
import math
from pathlib import Path
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlaxCompatibleGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.ir = nn.Linear(input_size, hidden_size, bias=True)
        self.iz = nn.Linear(input_size, hidden_size, bias=True)
        self.in_proj = nn.Linear(input_size, hidden_size, bias=True)

        self.hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hn = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.ir(x) + self.hr(h))
        z = torch.sigmoid(self.iz(x) + self.hz(h))
        n = torch.tanh(self.in_proj(x) + r * self.hn(h))
        new_h = (1.0 - z) * n + z * h
        return new_h


class TorchPolicyGRU(nn.Module):
    def __init__(
        self,
        action_dim: int,
        std_dev: float,
        obs_encoding_dim: int,
        gru_hidden_dim: int,
        gru_obs_combine_method: str,
        share_gru_obs_encoder: bool,
        policy_observation_indices: Sequence[int],
    ):
        super().__init__()

        if gru_obs_combine_method not in ("concat", "film"):
            raise ValueError(f"Unsupported gru_obs_combine_method: {gru_obs_combine_method}")

        self.action_dim = int(action_dim)
        self.obs_encoding_dim = int(obs_encoding_dim)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.gru_obs_combine_method = gru_obs_combine_method
        self.share_gru_obs_encoder = bool(share_gru_obs_encoder)

        indices = torch.as_tensor(policy_observation_indices, dtype=torch.long)
        self.register_buffer("policy_observation_indices", indices, persistent=False)

        policy_obs_dim = int(indices.numel())

        self.gru_obs_encoder_dense = nn.Linear(policy_obs_dim, self.obs_encoding_dim)
        self.gru_obs_encoder_ln = nn.LayerNorm(self.obs_encoding_dim, eps=1e-6)

        if not self.share_gru_obs_encoder:
            self.obs_encoder_dense = nn.Linear(policy_obs_dim, self.obs_encoding_dim)
            self.obs_encoder_ln = nn.LayerNorm(self.obs_encoding_dim, eps=1e-6)

        self.gru = FlaxCompatibleGRUCell(self.obs_encoding_dim, self.gru_hidden_dim)
        self.gru_ln = nn.LayerNorm(self.gru_hidden_dim, eps=1e-6)

        if self.gru_obs_combine_method == "film":
            self.gru_film_gamma = nn.Linear(self.gru_hidden_dim, self.obs_encoding_dim)
            self.gru_film_beta = nn.Linear(self.gru_hidden_dim, self.obs_encoding_dim)
            torso_in_dim = self.obs_encoding_dim
        else:
            torso_in_dim = self.obs_encoding_dim + self.gru_hidden_dim

        self.torso_dense1 = nn.Linear(torso_in_dim, 512)
        self.torso_ln1 = nn.LayerNorm(512, eps=1e-6)
        self.torso_dense2 = nn.Linear(512, 256)
        self.torso_dense3 = nn.Linear(256, 128)

        self.mean_head = nn.Linear(128, self.action_dim)
        self.policy_logstd = nn.Parameter(
            torch.full((1, self.action_dim), float(math.log(std_dev)), dtype=torch.float32)
        )

    @classmethod
    def from_meta(cls, meta: dict) -> "TorchPolicyGRU":
        return cls(
            action_dim=meta["action_dim"],
            std_dev=meta.get("std_dev", 1.0),
            obs_encoding_dim=meta["obs_encoding_dim"],
            gru_hidden_dim=meta["gru_hidden_dim"],
            gru_obs_combine_method=meta["gru_obs_combine_method"],
            share_gru_obs_encoder=meta["share_gru_obs_encoder"],
            policy_observation_indices=meta["policy_observation_indices"],
        )

    def initialize_carry(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, self.gru_hidden_dim, dtype=torch.float32, device=device)

    def _select_policy_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.index_select(dim=-1, index=self.policy_observation_indices)

    def obs_encode(self, obs: torch.Tensor) -> torch.Tensor:
        x = self._select_policy_obs(obs)
        x = self.obs_encoder_dense(x)
        x = self.obs_encoder_ln(x)
        x = F.elu(x)
        return x

    def gru_obs_encode(self, obs: torch.Tensor) -> torch.Tensor:
        x = self._select_policy_obs(obs)
        x = self.gru_obs_encoder_dense(x)
        x = self.gru_obs_encoder_ln(x)
        x = F.elu(x)
        return x

    def decode(self, obs_latent: torch.Tensor, gru_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gru_latent = self.gru_ln(gru_latent)
        gru_latent = F.elu(gru_latent)

        if self.gru_obs_combine_method == "concat":
            torso_in = torch.cat([obs_latent, gru_latent], dim=-1)
        else:
            gamma = self.gru_film_gamma(gru_latent)
            beta = self.gru_film_beta(gru_latent)
            torso_in = obs_latent * gamma + beta

        h = self.torso_dense1(torso_in)
        h = self.torso_ln1(h)
        h = F.elu(h)
        h = self.torso_dense2(h)
        h = F.elu(h)
        h = self.torso_dense3(h)
        h = F.elu(h)

        mean = self.mean_head(h)
        logstd = self.policy_logstd.expand(mean.shape[0], -1)
        return mean, logstd

    def forward_step(self, obs: torch.Tensor, carry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        squeeze_obs = False
        squeeze_carry = False

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze_obs = True
        if carry.dim() == 1:
            carry = carry.unsqueeze(0)
            squeeze_carry = True

        gru_obs_latent = self.gru_obs_encode(obs)
        next_carry = self.gru(gru_obs_latent, carry)

        if self.share_gru_obs_encoder:
            obs_latent = gru_obs_latent
        else:
            obs_latent = self.obs_encode(obs)

        mean, logstd = self.decode(obs_latent, next_carry)

        if squeeze_obs:
            mean = mean.squeeze(0)
            logstd = logstd.squeeze(0)
        if squeeze_carry:
            next_carry = next_carry.squeeze(0)

        return mean, logstd, next_carry

    def forward(self, obs: torch.Tensor, carry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, _, next_carry = self.forward_step(obs, carry)
        return mean, next_carry


def load_policy_from_files(
    weights_path: str | Path,
    meta_path: str | Path,
    device: torch.device,
) -> tuple[TorchPolicyGRU, dict]:
    weights_path = Path(weights_path)
    meta_path = Path(meta_path)

    with meta_path.open("r") as f:
        meta = json.load(f)

    model = TorchPolicyGRU.from_meta(meta).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, meta
