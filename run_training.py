import subprocess
import time
from nn_client import Client
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

FIELD_X_MIN, FIELD_X_MAX = -7.5, 0.0  # Own half
FIELD_Y_MIN, FIELD_Y_MAX = -4.5, 4.5  # Full lateral width
CENTER_CIRCLE_RADIUS = 1.5

class StrikerPlacementPolicy(nn.Module):
    def __init__(self, action_dim=3, hidden_dim=64):
        super().__init__()
        # Single constant input state since ball is always at (0,0)
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x):
        features = self.net(x)
        mean = self.mean_head(features)
        std = torch.clamp(self.log_std, -20, 2).exp().expand_as(mean)
        return mean, std

    def get_action(self, device):
        dummy_input = torch.ones(1, 1, device=device)
        mean, std = self.forward(dummy_input)
        dist = Normal(mean, std)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        return raw_action.squeeze(0), log_prob.squeeze(0)

def project_to_legal_pose(raw_action):
    """Maps unconstrained neural net outputs [-inf, inf] to valid field poses."""
    tanh_act = torch.tanh(raw_action).cpu().numpy()
    
    # 1. Scale to own half bounds
    x = (tanh_act[0] + 1) / 2 * (FIELD_X_MAX - FIELD_X_MIN) + FIELD_X_MIN
    y = tanh_act[1] * FIELD_Y_MAX
    theta = tanh_act[2] * np.pi

    # 2. Enforce Center Circle Exclusion geometrically
    dist_to_center = np.sqrt(x**2 + y**2)
    if dist_to_center < CENTER_CIRCLE_RADIUS:
        scale = (CENTER_CIRCLE_RADIUS + 0.05) / (dist_to_center + 1e-6)
        x *= scale
        y *= scale

    return (x, y, np.rad2deg(theta))

num_episodes=500
lr=1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = StrikerPlacementPolicy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=lr)

for episode in range(num_episodes):
    # 1. Sample continuous action from policy
    raw_action, log_prob = policy.get_action(device)
    
    # 2. Map to valid (x, y, theta) pose
    striker_pose = project_to_legal_pose(raw_action)
    print(f"\n[Episode {episode+1}] Teleporting Striker to Pose: x={striker_pose[0]:.2f}m, y={striker_pose[1]:.2f}m, theta={striker_pose[2]:.1f}°")

    # 3. Run Simulation Execution
    process = subprocess.Popen([
        "rcssservermj",
        "--host", "127.0.0.1",
        "--aport", "60000",
        "--mport", "60001",
        "--field", "hsl_m_26"
    ])
    #print(f"Server started in background with PID: {process.pid}")

    time.sleep(5)
    # TODO: change the initial position everytime
    client = Client(
        host='127.0.0.1',
        port=60000,
        team='Test',
        player_no=1,
        model_name='T1',
        start_position=striker_pose
    )
    client.run()
    #print(client.elapsed_time) # TODO write this into a file
    with open("training_log.txt", "a") as f:
        f.write(f"Run {i+1:>3} | elapsed_time: {client.elapsed_time:.4f}s\n")
    if client.elapsed_time > 0.0:
        reward = 60.0 - client.elapsed_time
        print(f"--> Success! Reached ball in {client.elapsed_time:.2f}s | Reward: {reward:.2f}")
    else:
        reward = -10.0
        print("--> Failed to reach ball within timeout | Reward: -10.0")

    loss = -log_prob * reward

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    time.sleep(5)
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("Server did not stop in time. Forcing kill...")
        process.kill()
        process.wait()
        
    print("Server stopped.")