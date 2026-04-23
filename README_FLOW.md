
# Physics-Guided Conditional Flow Matching on top of `unrolled_primal_dual_networks`

This bundle adds an end-to-end flow model to the original repo while reusing its core pieces:

- `lensless/model.py` for the deterministic unrolled primal-dual backbone
- `lensless/operators.py` for the FFT-based lensless forward/adjoint operator
- `lensless/diffusercam.py` for dataset loading
- `lensless/unet.py` for the velocity network backbone
- `lensless/training.py` / `lensless/evaluate.py` design pattern for training and checkpoint loading

## New files

- `lensless/flow.py`
- `lensless/flow_training.py`
- `experiment_flow.py`

## Core model

The final physics-aware velocity is

$v(x, t, y) = u_\theta(x, t, y) - \lambda(t) H^*(H x - y)$

where `u_theta` is learned and the second term is an analytic data-consistency drift.

Training uses straight-path conditional flow matching. If the target velocity is `(x1 - x0)`,
the target for the learned free velocity is

u_target = (x1 - x0) + lambda(t) H*(H x_t - y)

so that the composed physics-aware velocity matches the flow-matching target.

## Example commands

Train:
```bash
python experiment_flow.py /Datasets/lensless_learning/ \
  --backbone-model learned-primal-dual-and-five-models \
  --backbone-checkpoint weights/image_optimizer.ckpt \
  --lambda-max 0.5 \
  --sample-steps 40
```

Evaluate:
```bash
python experiment_flow.py /Datasets/lensless_learning/ \
  --eval \
  --checkpoint logs/physics_guided_flow/version_0/checkpoints/last.ckpt \
  --backbone-model learned-primal-dual-and-five-models \
  --backbone-checkpoint weights/image_optimizer.ckpt \
  --sample-steps 40 \
  --sample-method heun
```
