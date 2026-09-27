# Implementation and provenance

Reference for adapting the [structural RNN](structural-rnn.md). Use the [journal and conference citations](../README.md#citation) when applying the research methods.

## Provenance

Git history and the [MIT license](../LICENSE) identify Youngjoo Kim as the software author. The papers credit Vemula, Muelling, and Oh's [Social Attention implementation](https://www.cs.cmu.edu/~jeanoh/16-785/papers/vemula-icra2018-socialattention.pdf) and Jain et al.'s Structural-RNN approach; retain existing source notices and credit those underlying methods where used.

The [dataset guide](../dataset/Santander/README.md) describes the supplied road graphs and speed matrices. [CITATION.cff](../CITATION.cff) records the publications and repository separately.

## Differences from the published experiments

| Component | Source configuration |
|---|---|
| Spatial links | One directed spatial edge per adjacency entry equal to `1`; the journal uses directional links and the earlier conference model uses opposing pairs. |
| Optimization | Adagrad with library defaults; the papers specify Adam. Parsed learning-rate and decay arguments do not affect the active optimizer. |
| Data split | A 3:1 split of CSV rows; the journal describes a calendar split. |
| Epoch summary | The log footer selects the best observed epoch; the journal averages ten epochs. |
| Evaluation metric | Mean of per-sequence RMSE across nodes, rather than a root after pooling all squared errors. |

## Training and evaluation

Normal-case training resets batch pointers each epoch. Cross-dataset training reinitializes network parameters after its first epoch while retaining accumulated optimizer state; select the runner that matches the intended experiment.

Each optimizer step processes one sequence. `batch_size` groups sequences for reporting; training uses the final prediction in each non-overlapping `L+1`-row window. Each split must supply at least one complete batch.

Training enables dropout; evaluation disables dropout and gradient tracking. The model uses CPU tensors and the graph/data contracts in the [running guide](running.md). Record graph files, split, effective settings, and seeds with new measurements.

## Checks

The checked CPU environment uses Python 3.12, PyTorch 2.14, NumPy 2.5, and pandas 3.0. The [synthetic example](../example_synthetic.py) checks feature and output shapes. [Training regression checks](../tests/integration/training/README.md) cover exact full batches, epoch resets, and evaluation mode.
