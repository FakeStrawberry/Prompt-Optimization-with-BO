import math
from typing import Optional

import numpy as np


def _matern52_kernel(x1: np.ndarray, x2: np.ndarray, lengthscale: float = 0.5) -> np.ndarray:
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    diff = (x1[:, None, :] - x2[None, :, :]) / max(lengthscale, 1e-6)
    r = np.sqrt(np.sum(diff * diff, axis=-1))
    s5 = math.sqrt(5.0)
    return (1.0 + s5 * r + 5.0 * r * r / 3.0) * np.exp(-s5 * r)


def _simple_gp_predict(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray
) -> tuple:
    train_x = np.asarray(train_x, dtype=float)
    test_x = np.asarray(test_x, dtype=float)
    y = np.asarray(train_y, dtype=float)
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    if y_std < 1e-8:
        y_std = 1.0
    yn = (y - y_mean) / y_std
    kxx = _matern52_kernel(train_x, train_x) + 1e-5 * np.eye(len(train_x))
    kxs = _matern52_kernel(train_x, test_x)
    kss_diag = np.ones(test_x.shape[0])
    try:
        alpha = np.linalg.solve(kxx, yn)
        v = np.linalg.solve(kxx, kxs)
    except np.linalg.LinAlgError:
        kxx = kxx + 1e-3 * np.eye(len(train_x))
        alpha = np.linalg.solve(kxx, yn)
        v = np.linalg.solve(kxx, kxs)
    mean = kxs.T.dot(alpha) * y_std + y_mean
    var = np.maximum(kss_diag - np.sum(kxs * v, axis=0), 1e-9) * (y_std**2)
    return mean, np.sqrt(var)


def _normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def cross_validate_mse(
    z: np.ndarray, y: np.ndarray, seed: int = 0, use_botorch: bool = False
) -> float:
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2:
        return float("inf")

    rng = np.random.RandomState(seed)
    if n < 10:
        folds = [np.array([i]) for i in range(n)]
    else:
        indices = np.arange(n)
        rng.shuffle(indices)
        folds = [fold for fold in np.array_split(indices, 10) if len(fold)]

    preds = np.zeros(n, dtype=float)
    for test_idx in folds:
        train_idx = np.asarray([i for i in range(n) if i not in set(test_idx)], dtype=int)
        if len(train_idx) < 2:
            preds[test_idx] = float(np.mean(y[train_idx])) if len(train_idx) else float(np.mean(y))
            continue
        if use_botorch:
            mean = _botorch_predict(z[train_idx], y[train_idx], z[test_idx])
        else:
            mean, _ = _simple_gp_predict(z[train_idx], y[train_idx], z[test_idx])
        preds[test_idx] = mean
    return float(np.mean((preds - y) ** 2))


def optimize_targets(
    z: np.ndarray,
    y: np.ndarray,
    q: int,
    seed: int,
    num_restarts: int = 20,
    raw_samples: int = 512,
    strict_paper: bool = False,
) -> np.ndarray:
    """Select q feature targets.

    In strict mode this uses the paper's BoTorch SingleTaskGP + qLogNEI settings.
    Otherwise it falls back to a dependency-light random-candidate EI approximation,
    which is useful for smoke tests on machines without torch/botorch.
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    if z.ndim != 2 or z.shape[1] == 0:
        raise ValueError("Embedding matrix must have shape n x d with d > 0")

    try:
        return _optimize_targets_botorch(z, y, q, num_restarts, raw_samples)
    except Exception:
        if strict_paper:
            raise
        return _optimize_targets_fallback(z, y, q, seed, raw_samples=max(raw_samples, 512))


def _optimize_targets_botorch(
    z: np.ndarray, y: np.ndarray, q: int, num_restarts: int, raw_samples: int
) -> np.ndarray:
    import torch
    from botorch.optim import optimize_acqf

    try:
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    except Exception:
        from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement as qLogNoisyExpectedImprovement

    train_x = torch.tensor(z, dtype=torch.double)
    d = z.shape[1]
    model = _fit_botorch_model(z, y)
    acq = qLogNoisyExpectedImprovement(model=model, X_baseline=train_x)
    bounds = torch.stack(
        [torch.zeros(d, dtype=torch.double), torch.ones(d, dtype=torch.double)]
    )
    candidates, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    return candidates.detach().cpu().numpy()


def _fit_botorch_model(z: np.ndarray, y: np.ndarray):
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize, Standardize
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.mlls import ExactMarginalLogLikelihood

    train_x = torch.tensor(z, dtype=torch.double)
    train_y = torch.tensor(y, dtype=torch.double).unsqueeze(-1)
    d = z.shape[1]
    covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))
    model = SingleTaskGP(
        train_x,
        train_y,
        covar_module=covar_module,
        input_transform=Normalize(d=d),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def _botorch_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    import torch

    model = _fit_botorch_model(train_x, train_y)
    model.eval()
    test_tensor = torch.tensor(test_x, dtype=torch.double)
    with torch.no_grad():
        posterior = model.posterior(test_tensor)
    return posterior.mean.detach().cpu().numpy().reshape(-1)


def _optimize_targets_fallback(
    z: np.ndarray, y: np.ndarray, q: int, seed: int, raw_samples: int
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    d = z.shape[1]
    candidates = rng.rand(raw_samples, d)
    chosen = []
    train_x = np.asarray(z, dtype=float)
    train_y = np.asarray(y, dtype=float)
    best = float(np.max(train_y))
    for _ in range(q):
        mean, std = _simple_gp_predict(train_x, train_y, candidates)
        std = np.maximum(std, 1e-9)
        imp = mean - best
        u = imp / std
        ei = imp * _normal_cdf(u) + std * _normal_pdf(u)
        if chosen:
            # Small diversity penalty so the fallback behaves like a batch selector.
            dist = np.min(
                np.linalg.norm(candidates[:, None, :] - np.asarray(chosen)[None, :, :], axis=-1),
                axis=1,
            )
            ei = ei * (0.25 + dist)
        idx = int(np.argmax(ei))
        point = candidates[idx].copy()
        chosen.append(point)
        train_x = np.vstack([train_x, point])
        train_y = np.append(train_y, best)
        candidates[idx] = rng.rand(d)
    return np.asarray(chosen, dtype=float)
