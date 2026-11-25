"""Run a tiny record-aware TSFlow training-contract smoke test on CPU."""

import torch
from torch.utils.data import DataLoader, Subset

from sequence_data import default_cwru_dataset_dir
from tsflow_data import TSFlowManifestDataset
from tsflow.model import TSFlowCond


CONTEXT_LENGTH = 96
PREDICTION_LENGTH = 48


def create_model():
    return TSFlowCond(
        setting="univariate",
        target_dim=1,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        backbone_params={
            "input_dim": 1,
            "output_dim": 1,
            "step_emb": 16,
            "num_residual_blocks": 1,
            "residual_block": "s4",
            "hidden_dim": 16,
            "dropout": 0.0,
            "init_skip": False,
            "feature_skip": True,
        },
        prior_params={
            "kernel": "ou",
            "gamma": 1.0,
            "context_freqs": 2,
        },
        optimizer_params={"lr": 1e-3},
        ema_params={"beta": 0.999, "update_after_step": 10, "update_every": 1},
        frequency="H",
        normalization="identity",
        use_lags=False,
        use_ema=False,
        num_steps=2,
        solver="euler",
        matching="random",
    )


def main():
    torch.manual_seed(42)
    dataset = TSFlowManifestDataset(
        default_cwru_dataset_dir(),
        split="train",
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        stride=64,
    )
    subset = Subset(dataset, range(min(8, len(dataset))))
    batch = next(iter(DataLoader(subset, batch_size=4, shuffle=False)))

    model = create_model().cpu().train()
    x1, x0, _, _, _, features = model._extract_features(batch)
    time = torch.rand((x1.shape[0], 1))
    loss = model.p_losses(x1, x0, time, features)
    loss.backward()

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite Flow Matching loss: {loss.item()}")
    gradient_count = sum(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    if gradient_count == 0:
        raise RuntimeError("No finite gradients were produced")

    print(f"records={dataset.record_ids}")
    print(
        f"past={tuple(batch['past_target'].shape)}, future={tuple(batch['future_target'].shape)}"
    )
    print(
        f"x0={tuple(x0.shape)}, x1={tuple(x1.shape)}, features={tuple(features.shape)}"
    )
    print(f"loss={loss.item():.6f}, parameters_with_finite_gradients={gradient_count}")
    print("TSFlow manifest adapter smoke test passed")


if __name__ == "__main__":
    main()
