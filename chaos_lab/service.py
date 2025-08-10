from fastapi import FastAPI
from pydantic import BaseModel

from .sim import henon, lorenz

app = FastAPI(title="Neural Chaos Lab API")


@app.get("/health")
def health():
    return {"status": "ok"}


class SimReq(BaseModel):
    system: str = "lorenz"
    n: int = 10000


@app.post("/simulate")
def simulate(req: SimReq):
    if req.system == "lorenz":
        arr = lorenz(n=req.n)
        cols = ["x", "y", "z"]
    else:
        arr = henon(n=req.n)
        cols = ["x", "y"]
    return {"columns": cols, "data": arr[:1000].tolist()}
