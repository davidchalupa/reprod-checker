# toy_train.py
import torch
import torch.nn as nn
import torch.optim as optim

def train_fn(cfg):
    # cfg: {seed, device, run_idx}
    device = cfg.get("device", "cpu")
    seed = cfg.get("seed", 0)

    # local seed (common in user code)
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # tiny model + dataset
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
    model.to(device)
    opt = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    X = torch.randn(20, 10, device=device)
    y = torch.randn(20, 1, device=device)

    model.train()
    for epoch in range(3):
        opt.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_out = model(X)
        val_loss = float(loss_fn(val_out, y).item())

    # return model and metrics
    return model, {"val_loss": val_loss}
