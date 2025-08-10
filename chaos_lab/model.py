import numpy as np, torch, torch.nn as nn

class LSTMForecast(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=2, out_dim=None):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, out_dim or in_dim)
    def forward(self, x):
        y,_ = self.lstm(x)
        return self.head(y[:,-1])

def make_windows(arr, win=50, horizon=1):
    X=[]; Y=[]
    for i in range(len(arr)-win-horizon+1):
        X.append(arr[i:i+win])
        Y.append(arr[i+win+horizon-1])
    return np.array(X), np.array(Y)

def train_model(series, win=50, horizon=1, epochs=10, lr=1e-3):
    X,Y = make_windows(series, win, horizon)
    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    model = LSTMForecast(in_dim=series.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(X_t)
        loss = lossf(pred, Y_t)
        loss.backward(); opt.step()
    return model

def predict(model, context, steps=10):
    import torch
    x = torch.tensor(context[-50:], dtype=torch.float32).unsqueeze(0)
    outs=[]
    cur=context.copy()
    for _ in range(steps):
        y=model(x).detach().numpy()[0]
        outs.append(y)
        cur=np.vstack([cur, y])
        x=torch.tensor(cur[-50:], dtype=torch.float32).unsqueeze(0)
    return np.array(outs)
