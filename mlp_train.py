import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


import joblib

df_raw=pd.read_csv("logs/simulation_log.csv")
df=pd.get_dummies(df_raw,"Slice")

X=df[["Slice_eMBB","Slice_URLLC","Slice_mMTC","Asked","Admitted","Dropped"]]
y=df["SLA_violation"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=20)

#normalizing
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

#to tnsors:
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1,1)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim,128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,8),
            nn.ReLU(),
            nn.Linear(8,1),
            nn.Sigmoid())

    def forward(self,x):
        return self.model(x)

model= MLPClassifier(input_dim=X_train_tensor.shape[1])
criterion=nn.BCELoss()
optimizer=optim.Adam(model.parameters(),lr=0.005)

for epoch in range(600):
    model.train()
    optimizer.zero_grad()
    outputs= model(X_train_tensor)
    loss = criterion(outputs,y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10==0:
        print(f"Epoch{epoch}: Loss={loss.item():0.4f}")

model.eval()
with torch.no_grad():
    preds = model(X_test_tensor)
    preds_label = (preds > 0.5).float()

print(confusion_matrix(y_test_tensor, preds_label))
print(classification_report(y_test_tensor, preds_label))

torch.save(model.state_dict(), "models/mlp_model.pth")
