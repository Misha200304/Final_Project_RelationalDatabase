import pandas as pd
import torch
import torch.nn as nn

df = pd.read_csv("correct_master.csv")

required_cols = ["b365w", "b365l", "rank_diff", "surface", "tourney_level"]
df = df.dropna(subset=required_cols).copy()

df_encoded = pd.get_dummies(df, columns=["surface", "tourney_level"])

surface_cols = sorted([c for c in df_encoded.columns if c.startswith("surface_")])
level_cols   = sorted([c for c in df_encoded.columns if c.startswith("tourney_level_")])
onehot_cols  = surface_cols + level_cols

rows = []
for _, row in df_encoded.iterrows():
    # Convert odds to implied probability BEFORE feeding to model
    imp_w = 1.0 / float(row["b365w"])
    imp_l = 1.0 / float(row["b365l"])

    base = {
        "imp_w":      imp_w,
        "imp_l":      imp_l,
        "rank_diff":  float(row["rank_diff"]),
        "target":     1.0,
    }
    for c in onehot_cols:
        base[c] = float(row[c])
    rows.append(base)

    flipped = {
        "imp_w":      imp_l,   # swapped
        "imp_l":      imp_w,   # swapped
        "rank_diff":  -float(row["rank_diff"]),
        "target":     0.0,
    }
    for c in onehot_cols:
        flipped[c] = float(row[c])
    rows.append(flipped)

train_df = pd.DataFrame(rows)

numeric_cols = ["imp_w", "imp_l", "rank_diff"]
feature_cols = numeric_cols + onehot_cols

features = torch.tensor(train_df[feature_cols].values, dtype=torch.float32)
target   = torch.tensor(train_df[["target"]].values,   dtype=torch.float32)

# Only normalize the numeric columns, not the one-hot columns
fm = torch.zeros(len(feature_cols))
fs = torch.ones(len(feature_cols))

for i, col in enumerate(numeric_cols):
    fm[i] = features[:, i].mean()
    fs[i] = features[:, i].std()
    if fs[i] == 0:
        fs[i] = 1.0

fm = fm.unsqueeze(0)
fs = fs.unsqueeze(0)

X = (features - fm) / fs
Y = target


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


model     = TennisNet(len(feature_cols))
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 250
for epoch in range(epochs):
    optimizer.zero_grad()       # zero BEFORE forward pass
    y_hat = model(X)
    loss  = criterion(y_hat, Y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}  Loss: {loss.item():.6f}")

torch.save({
    "fm":           fm,
    "fs":           fs,
    "feature_cols": feature_cols,
    "parameters":   model.state_dict(),
    "input_dim":    len(feature_cols),
}, "model.pth")

print("Saved model.pth")