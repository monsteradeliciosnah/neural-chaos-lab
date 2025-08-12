import pandas as pd
import typer

from .model import predict, train_model
from .sim import henon, lorenz

app = typer.Typer(help="Neural Chaos Lab CLI")


@app.command()
def generate(system: str = "lorenz", n: int = 10000, out: str = "data/series.csv"):
    import os

    os.makedirs("data", exist_ok=True)
    if system == "lorenz":
        arr = lorenz(n=n)
        cols = ["x", "y", "z"]
    else:
        arr = henon(n=n)
        cols = ["x", "y"]
    pd.DataFrame(arr, columns=cols).to_csv(out, index=False)
    print(out)


@app.command()
def train(
    data: str = "data/series.csv",
    win: int = 50,
    horizon: int = 1,
    epochs: int = 10,
    lr: float = 1e-3,
):
    df = pd.read_csv(data)
    m = train_model(df.values, win, horizon, epochs, lr)
    import os

    import torch

    os.makedirs("models", exist_ok=True)
    torch.save(m.state_dict(), "models/lstm.pt")
    print("models/lstm.pt")


@app.command()
def forecast(data: str = "data/series.csv", steps: int = 200):
    import torch

    from .model import LSTMForecast

    df = pd.read_csv(data)
    in_dim = df.values.shape[1]
    m = LSTMForecast(in_dim=in_dim)
    m.load_state_dict(torch.load("models/lstm.pt", map_location="cpu"))
    preds = predict(m, df.values, steps=steps)
    out = pd.DataFrame(preds, columns=[f"y{i}" for i in range(preds.shape[1])])
    out.to_csv("reports/forecast.csv", index=False)
    print("reports/forecast.csv")


if __name__ == "__main__":
    app()
