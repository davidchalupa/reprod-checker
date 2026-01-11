# toy_train.py
import random
import numpy as np
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


def train_fn_reproducible(cfg):
    """
    Minimal reproducible train_fn for the PoC.
    - ignores incoming seed variations by setting a fixed internal seed
    - enables PyTorch deterministic mode (best-effort)
    - uses CPU-friendly deterministic ops and num_workers=0 style behavior
    Returns: (model, {"val_loss": ...})
    """
    # enforce deterministic behavior (best-effort)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # older torch versions may not support this; it's best-effort
        pass

    # Reduce threading nondeterminism on some platforms
    torch.set_num_threads(1)

    # FIXED internal seed for a reproducible run (demo-only)
    FIXED_SEED = 0
    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FIXED_SEED)

    device = cfg.get("device", "cpu")

    # create a fixed (deterministic) small dataset
    # since we seeded above, these tensors will be identical across runs
    X = torch.randn(20, 10, device=device)
    y = torch.randn(20, 1, device=device)

    # tiny deterministic model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # short, deterministic training loop
    model.train()
    for epoch in range(3):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

    # validation metric (deterministic because the whole run was seeded)
    model.eval()
    with torch.no_grad():
        val_loss = float(loss_fn(model(X), y).item())

    # return model and metrics (fits the PoC ReproChecker contract)
    return model, {"val_loss": val_loss}
