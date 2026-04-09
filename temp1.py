# =====================================================
# Graph Attention Network for Heat Stress Prediction
# =====================================================

# !pip install torch torch-geometric pandas scikit-learn

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =====================================================
# 1. Load Dataset
# =====================================================

df = pd.read_csv("combined_dataset.csv")

FEATURES = [
    "body_temp","ambient_temp","humidity","heart_rate","skin_resistance",
    "resp_rate","movement","avg_sensor_temp","sensor_spread",
    "iaq","lux","sound",
    "temp_humidity_index","heat_index","hr_temp_product",
    "skin_resistance_normalized","body_amb_diff"
]

X = df[FEATURES].values

# ensure valid labels
y_heat = df["heat_stress_label"].astype(int).values
y_dehyd = df["dehydration_label"].astype(int).values

y_heat = np.clip(y_heat,0,2)
y_dehyd = np.clip(y_dehyd,0,1)


# =====================================================
# 2. Feature Scaling
# =====================================================

scaler = StandardScaler()
X = scaler.fit_transform(X)


# =====================================================
# 3. Build Graph using Correlations
# =====================================================

corr = np.corrcoef(X.T)

threshold = 0.25
edges = []

n = corr.shape[0]

for i in range(n):
    for j in range(n):
        if i != j and abs(corr[i,j]) > threshold:
            edges.append((i,j))

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()


# =====================================================
# 4. Convert Rows → Graph Objects
# =====================================================

graphs = []

for i in range(len(X)):

    node_features = torch.tensor(X[i], dtype=torch.float).view(-1,1)

    graph = Data(
        x=node_features,
        edge_index=edge_index,
        y_heat=torch.tensor(y_heat[i], dtype=torch.long),
        y_dehyd=torch.tensor(y_dehyd[i], dtype=torch.long)
    )

    graphs.append(graph)


# =====================================================
# 5. Train/Test Split
# =====================================================

train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)

train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=64)


# =====================================================
# 6. Graph Attention Model
# =====================================================

class PhysioGAT(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.gat1 = GATConv(1, 64, heads=8, dropout=0.3)
        self.gat2 = GATConv(64*8, 32, heads=4, dropout=0.3)

        self.fc = Linear(32*4, 16)
        self.bn = BatchNorm1d(16)

        self.heat_head = Linear(16,3)
        self.dehyd_head = Linear(16,2)

    def forward(self,x,edge_index,batch):

        x = self.gat1(x,edge_index)
        x = F.relu(x)

        x = self.gat2(x,edge_index)
        x = F.relu(x)

        x = global_mean_pool(x,batch)

        x = self.fc(x)
        x = self.bn(x)
        x = F.relu(x)

        heat = self.heat_head(x)
        dehyd = self.dehyd_head(x)

        return heat,dehyd


# =====================================================
# 7. Force CPU (avoids CUDA errors)
# =====================================================

device = torch.device("cpu")

model = PhysioGAT().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# =====================================================
# 8. Training
# =====================================================

epochs = 40

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for batch in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        heat_pred,dehyd_pred = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        loss_heat = F.cross_entropy(heat_pred, batch.y_heat)
        loss_dehyd = F.cross_entropy(dehyd_pred, batch.y_dehyd)

        loss = loss_heat + loss_dehyd

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss {total_loss:.4f}")


# =====================================================
# 9. Evaluation
# =====================================================

model.eval()

heat_true=[]
heat_pred=[]

dehyd_true=[]
dehyd_pred=[]

with torch.no_grad():

    for batch in test_loader:

        batch = batch.to(device)

        heat,dehyd = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        heat_p = heat.argmax(dim=1)
        dehyd_p = dehyd.argmax(dim=1)

        heat_true.extend(batch.y_heat.cpu().numpy())
        heat_pred.extend(heat_p.cpu().numpy())

        dehyd_true.extend(batch.y_dehyd.cpu().numpy())
        dehyd_pred.extend(dehyd_p.cpu().numpy())


print("\n===== RESULTS =====")

print("Heat Stress Accuracy:", accuracy_score(heat_true,heat_pred))
print("Dehydration Accuracy:", accuracy_score(dehyd_true,dehyd_pred))



# ===== RESULTS =====
# Heat Stress Accuracy: 0.8412643278916291
# Dehydration Accuracy: 0.904480722473081