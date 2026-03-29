import torch
import torch.nn as nn


class TennisNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


model_data   = torch.load("model.pth", map_location="cpu")
fm           = model_data["fm"]
fs           = model_data["fs"]
feature_cols = model_data["feature_cols"]
input_dim    = model_data["input_dim"]

model = TennisNet(input_dim)
model.load_state_dict(model_data["parameters"])
model.eval()

# --- Input your odds here ---
b365_player1 = 1.95
b365_player2 = 1.95
rank_diff     = -12.0   # player1_rank minus player2_rank (negative = player1 is higher ranked)
surface       = "Hard"  # Hard / Clay / Grass / Carpet
tourney_level = "M"     # M / G / A / F / D

# Convert odds to implied probability
values = {col: 0.0 for col in feature_cols}
values["imp_w"]     = 1.0 / b365_player1
values["imp_l"]     = 1.0 / b365_player2
values["rank_diff"] = float(rank_diff)

surface_key = f"surface_{surface}"
level_key   = f"tourney_level_{tourney_level}"

if surface_key in values:
    values[surface_key] = 1.0
else:
    print(f"Warning: '{surface_key}' not found in model features — treated as unknown surface")

if level_key in values:
    values[level_key] = 1.0
else:
    print(f"Warning: '{level_key}' not found in model features — treated as unknown level")

features = torch.tensor([[values[col] for col in feature_cols]], dtype=torch.float32)
features = (features - fm) / fs

with torch.no_grad():
    logit = model(features)
    p1    = torch.sigmoid(logit).item()

p2 = 1.0 - p1

print(f"Player 1 win probability: {p1 * 100:.1f}%")
print(f"Player 2 win probability: {p2 * 100:.1f}%")