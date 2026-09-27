# Provenance and reproducibility limits

Evidence and boundaries for using [SRNN](../README.md) as a technical reference. The historical source baseline is revision [a9ccd8b2117c79106aa4dc1ffb9c21e31fdf7b11](https://github.com/rhymesg/SRNN/tree/a9ccd8b2117c79106aa4dc1ffb9c21e31fdf7b11).

## Provenance

- Git history, the original README, and the MIT copyright identify Youngjoo Kim as the software author; the repository accompanies the [journal and preliminary conference papers](../README.md#citation).
- The journal's Section IV-A2 and conference paper's Section 3.2 state that the implementation builds on the PyTorch implementation of Vemula, Muelling, and Oh's [*Social Attention: Modeling Attention in Human Crowds*](https://www.cs.cmu.edu/~jeanoh/16-785/papers/vemula-icra2018-socialattention.pdf) (their reference [11]).
- The papers also credit Jain et al.'s *Structural-RNN* for the underlying approach; this traffic-specific adaptation is not presented here as the original general SRNN implementation.
- The Social Attention paper identifies [vvanirudh/srnn-pytorch](https://github.com/vvanirudh/srnn-pytorch) as its implementation; that URL returned HTTP 404 when checked on 2026-09-27. This historical lead does not identify the revision inherited by SRNN; reused code and associated notices remain unverified.
- Existing author notices and the [MIT license](../LICENSE) are retained; dataset provenance gaps are recorded in the [dataset guide](../dataset/Santander/README.md#preprocessing-and-provenance-limits).
- No software release tag or software DOI was present in the inspected checkout; [CITATION.cff](../CITATION.cff) keeps paper DOIs in publication entries rather than assigning one to the software.

## Differences from the published experiments

| Topic | Publication | Supplied implementation |
|---|---|---|
| Spatial links | Journal Section III-B uses directional links; ICASSP Section 2.2 uses opposing pairs | One spatial edge per adjacency entry equal to `1`; reverse edges are not inserted |
| Optimizer | Journal Section IV-A2 and ICASSP Section 3.2 specify Adam | Both active training functions use Adagrad with library defaults |
| Learning-rate schedule | Journal Table I gives a starting rate and exponential decay | Parsed rate/decay options are unused; no scheduler |
| Data length/split | Journal Section IV-A1 describes 35,040 readings and a calendar split | Bundled files have 33,504 rows; loader uses a 3:1 row split |
| Epoch summaries | Journal Section IV-B averages ten epochs | Default run requests one epoch; log footer selects the best observed epoch |
| RMSE | Journal Eq. (12) takes a root after pooling squared errors across evaluation values | Code takes RMSE across nodes per sequence, then averages those RMSEs |

The default hidden and embedding sizes match journal Table I; the ICASSP model used larger sizes. A constant trainable parameter count does not imply constant runtime or state memory across graph sizes.

## Training and evaluation limitations

- `Run_SRNN_NormalCase` resets pointers each epoch; exact full batches are accepted.
- `Run_SRNN_Different_Dataset` reloads data each phase, but calls `net.initialize()` after the first epoch while retaining the optimizer's accumulated state; it does not perform ordinary continuous multi-epoch training.
- Both runners now use training/evaluation modes and disable gradient tracking during evaluation forward calls.
- `batch_size` groups sequences for reporting; gradients are cleared and the optimizer steps for each sequence, rather than one accumulated minibatch update.
- Training and evaluation use only the last prediction in each non-overlapping `L+1`-row window; intervening outputs do not contribute directly to the loss.
- Splits containing zero complete batches lead to division by zero, as detailed in the [running guide](running.md#sequence-boundaries-and-data-requirements).
- `--num_layer` and `--lambda_param` are unused, alongside the rate/decay settings above; no CLI seed is supplied.
- CPU tensors are constructed internally; moving only the model to a GPU does not provide a working GPU path.
- Isolated nodes, nonzero adjacency diagonals, missing values, and mismatched graph/data sizes are not supported robustly; see the [input contract](../dataset/Santander/README.md#input-contract).
- Dataset 5, pretrained checkpoints, baseline CNN/CapsNet implementations, raw-data preprocessing, and scripts reproducing the publication tables are absent.

Optimizer settings, transfer-training reinitialization, and metric aggregation remain research choices; the boundary, epoch-reset, and evaluation-mode fixes do not reconcile those choices with the papers.

## Verification status

Checks on 2026-09-27 used a temporary CPU environment on macOS arm64 with Python 3.12.14, PyTorch 2.14.0, NumPy 2.5.3, and pandas 3.0.6. The historical README's Python 3.5 environment has not been reconstructed, and no broader compatibility matrix is claimed.

- Checked bundled CSV dimensions, finite speed values, binary adjacency values, zero diagonals, and absence of isolated nodes.
- Passed the synthetic forward check and the README's small training/evaluation command, including checkpoint and CSV inspection; these are execution checks, not scientific validation.
- The small training run reported 87,905 trainable parameters and emitted NumPy deprecation warnings at the tensor-to-array conversion in `loss_RMSE`; it completed successfully.
- Regression checks cover exact-full-batch splits, pointer resets, and two-epoch normal training.
- Checked citation metadata against the papers and institutional/arXiv records, and validated `CITATION.cff` against the CFF 1.2.0 schema.
- Checked documentation links and Python syntax; the `SRNN.forward` docstring now describes the input tensors used by the caller.
- Full-dataset training, published RMSE comparisons, cross-network accuracy, and GPU execution have not been reproduced.

For reproducible future experiments, record the exact commit, dependency versions, graph/data files, split, all effective settings, random seeds, and evaluation aggregation method. Resolve the discrepancies above before comparing new measurements with published tables.
