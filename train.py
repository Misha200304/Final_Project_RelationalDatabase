import pandas as pd
import torch
import torch.nn as nn

df = pd.read_csv("correct_master.csv")

required_cols = [
    "b365w", "b365l", "rank_diff", "surface", "tourney_level",
    "w_ace", "w_df", "w_1stin", "w_1stwon", "w_bpsaved", "w_bpfaced", "w_bp_conv_pct", "w_svpt", "w_2ndwon",
    "l_ace", "l_df", "l_1stin", "l_1stwon", "l_bpsaved", "l_bpfaced", "l_bp_conv_pct", "l_svpt", "l_2ndwon"
]
df = df.dropna(subset=required_cols).copy()

# Derived stats
df["w_1st_serve_return_won"] = df["l_1stin"] - df["l_1stwon"]
df["l_1st_serve_return_won"] = df["w_1stin"] - df["w_1stwon"]

df["w_2nd_serve_return_won"] = (df["l_svpt"] - df["l_df"] - df["l_1stin"]) - df["l_2ndwon"]
df["l_2nd_serve_return_won"] = (df["w_svpt"] - df["w_df"] - df["w_1stin"]) - df["w_2ndwon"]

df_encoded = pd.get_dummies(df, columns=["surface", "tourney_level"])

surface_cols = sorted([c for c in df_encoded.columns if c.startswith("surface_")])
level_cols   = sorted([c for c in df_encoded.columns if c.startswith("tourney_level_")])
onehot_cols  = surface_cols + level_cols

rows = []
for _, row in df_encoded.iterrows():
    imp_w = 1.0 / float(row["b365w"])
    imp_l = 1.0 / float(row["b365l"])

    # base: P1 is winner, P2 is loser
    base = {
        "p1_imp": imp_w, "p2_imp": imp_l, "rank_diff": float(row["rank_diff"]),
        "p1_ace": float(row["w_ace"]), "p2_ace": float(row["l_ace"]),
        "p1_df": float(row["w_df"]), "p2_df": float(row["l_df"]),
        "p1_1stin": float(row["w_1stin"]), "p2_1stin": float(row["l_1stin"]),
        "p1_1stwon": float(row["w_1stwon"]), "p2_1stwon": float(row["l_1stwon"]),
        "p1_bpsaved": float(row["w_bpsaved"]), "p2_bpsaved": float(row["l_bpsaved"]),
        "p1_bpfaced": float(row["w_bpfaced"]), "p2_bpfaced": float(row["l_bpfaced"]),
        "p1_bp_conv_pct": float(row["w_bp_conv_pct"]), "p2_bp_conv_pct": float(row["l_bp_conv_pct"]),
        "p1_1st_serve_return_won": float(row["w_1st_serve_return_won"]), "p2_1st_serve_return_won": float(row["l_1st_serve_return_won"]),
        "p1_2nd_serve_return_won": float(row["w_2nd_serve_return_won"]), "p2_2nd_serve_return_won": float(row["l_2nd_serve_return_won"]),
        "target": 1.0,
    }
    for c in onehot_cols:
        base[c] = float(row[c])
    rows.append(base)

    # flipped: P1 is loser, P2 is winner
    flipped = {
        "p1_imp": imp_l, "p2_imp": imp_w, "rank_diff": -float(row["rank_diff"]),
        "p1_ace": float(row["l_ace"]), "p2_ace": float(row["w_ace"]),
        "p1_df": float(row["l_df"]), "p2_df": float(row["w_df"]),
        "p1_1stin": float(row["l_1stin"]), "p2_1stin": float(row["w_1stin"]),
        "p1_1stwon": float(row["l_1stwon"]), "p2_1stwon": float(row["w_1stwon"]),
        "p1_bpsaved": float(row["l_bpsaved"]), "p2_bpsaved": float(row["w_bpsaved"]),
        "p1_bpfaced": float(row["l_bpfaced"]), "p2_bpfaced": float(row["w_bpfaced"]),
        "p1_bp_conv_pct": float(row["l_bp_conv_pct"]), "p2_bp_conv_pct": float(row["w_bp_conv_pct"]),
        "p1_1st_serve_return_won": float(row["l_1st_serve_return_won"]), "p2_1st_serve_return_won": float(row["w_1st_serve_return_won"]),
        "p1_2nd_serve_return_won": float(row["l_2nd_serve_return_won"]), "p2_2nd_serve_return_won": float(row["w_2nd_serve_return_won"]),
        "target": 0.0,
    }
    for c in onehot_cols:
        flipped[c] = float(row[c])
    rows.append(flipped)

train_df = pd.DataFrame(rows)

numeric_cols = [
    "p1_imp", "p2_imp", "rank_diff",
    "p1_ace", "p2_ace", "p1_df", "p2_df",
    "p1_1stin", "p2_1stin", "p1_1stwon", "p2_1stwon",
    "p1_bpsaved", "p2_bpsaved", "p1_bpfaced", "p2_bpfaced",
    "p1_bp_conv_pct", "p2_bp_conv_pct",
    "p1_1st_serve_return_won", "p2_1st_serve_return_won",
    "p1_2nd_serve_return_won", "p2_2nd_serve_return_won"
]
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