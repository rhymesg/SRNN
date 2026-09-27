# Running SRNN

Guide to the existing [training script](../main_SRNN.py) and [synthetic example](../example_synthetic.py). Follow the [installation and short-run commands](../README.md#installation) first.

## Entry points

- `main()` currently calls `Run_SRNN_NormalCase(args, no_dataset=1)`; dataset selection is not a CLI option.
- `Run_SRNN_Different_Dataset(args, train_id, eval_id)` trains on one graph and evaluates on another after replacing the graph and resetting sequence states.
- `Run_SRNN_Scalability(args)` calls that function for all pairs of datasets 1–4 and overrides the epoch count and data limit.
- Alternative experiment calls in `main()` are commented out; edit the selected call only after reviewing [training limitations](limitations.md#training-and-evaluation-limitations).
- `Run_SRNN_test_parameters` references dataset 5, which is absent; it is not a runnable supplied experiment.

List parser options from the repository root:

```bash
python main_SRNN.py --help
```

## Settings that affect execution

| Option | Meaning |
|---|---|
| `--numData_set` | Use a prefix of rows; `-1` uses all supplied rows |
| `--numData_train_set` | Training prefix length; `-1` splits at `floor(3*T/4)` |
| `--numNodes_set` | Number of leading data columns; the graph must have the same size |
| `--seq_length` | Number of observed steps per prediction; each stored sequence also contains one target row |
| `--batch_size` | Number of sequences grouped for iteration/logging; optimizer steps still occur per sequence |
| `--node_rnn_size`, `--edge_rnn_size` | Hidden/cell widths |
| `--node_embedding_size`, `--edge_embedding_size` | Feature/context embedding widths |
| `--dropout` | Embedding dropout probability, also active during the original evaluation loop |
| `--grad_clip` | Gradient norm threshold before each optimizer step |
| `--num_epochs` | Requested epochs; normal-case pointers are not reset between epochs |

`--learning_rate`, `--decay_rate`, `--lambda_param`, and `--num_layer` are parsed but not applied by the active training functions/model. Both training functions instantiate `Adagrad(net.parameters())` with library defaults; check [Adagrad documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adagrad.html) for the installed version.

Keep node input/output dimensions at `1` and edge input dimensions at `2` for this loader. `--numNodes_set` does not crop the adjacency matrix; selecting fewer columns alone fails the graph-size assertion.

## Sequence boundaries and data requirements

[DataLoader](../dataLoader.py) scales speed by `150`, splits in CSV row order, and creates non-overlapping sequences of `L+1` rows. For split size `S` and batch size `B`, it advertises `floor(floor(S/(L+1))/B)` batches and drops the remaining rows.

The batch readers additionally assert `idx + L + 2 < S`. This is stricter than the advertised batch count: some otherwise valid sizes fail on their last batch, and splits with zero batches cause division by zero in training/evaluation.

The README's short run uses sizes checked to satisfy these restrictions. It runs four training batches and one evaluation batch on dataset 1, performs eight optimizer steps, and saves one checkpoint.

## Outputs and randomness

- `save/dataset_<train_id>/srnn_model_epoch<epoch>.tar` contains `epoch`, `state_dict`, and `optimizer_state_dict`; it does not save args, graph, split, seed, or dependency versions.
- `log/_loss_eval_dataset_<id>.csv` records normal-case epoch/loss pairs; cross-graph logs use `SRNN_loss_eval_dataset_<train>_on_<eval>.csv`.
- The final CSV row repeats the best epoch and loss, rather than another epoch; a one-epoch run has two identical rows.
- Losses in logs are averages of per-sequence final-step RMSEs after inverse scaling; training backpropagates normalized final-step MSE.
- Existing files at these paths are overwritten; cross-graph experiments reuse checkpoint paths for the same training dataset.
- The training script does not set seeds and keeps dropout active in evaluation, so numerical logs vary between runs.
- The synthetic example sets a seed and disables dropout for its untrained forward check; its shape/finite checks do not require exact floating-point predictions.

The script produces console output, checkpoints, and CSV logs, but no plots or standalone prediction export. See [verification status](limitations.md#verification-status) for what has actually been executed.
