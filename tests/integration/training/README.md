# Training regression checks

From the repository root, in the README's PyTorch environment:

```bash
PYTHONPATH=. python tests/integration/training/verify_training.py
```

The test writes a 16-row, three-sensor synthetic dataset in a temporary directory. Each split contains exactly one full batch. It runs two epochs with training dropout, checks training/evaluation modes and gradient contexts, then verifies both checkpoints and finite evaluation logs. No bundled dataset or plots are needed. The check verifies execution, not forecast accuracy.
