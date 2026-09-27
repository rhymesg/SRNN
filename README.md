# SRNN

## Overview

PyTorch structural recurrent neural network (SRNN) for short-term traffic speed prediction from historical speed measurements and a road-network adjacency matrix.

This is Youngjoo Kim's research code accompanying the 2019 IEEE Sensors Journal paper **“Scalable Learning With a Structural Recurrent Neural Network for Short-Term Traffic Prediction”**, following the preliminary ICASSP paper. See the [canonical repository](https://github.com/rhymesg/SRNN), [citations and full texts](#citation), and [provenance](docs/limitations.md#provenance).

The model combines shared node LSTMs, spatial-edge LSTMs, and temporal-edge LSTMs for graph-based time-series forecasting. Shared weights and summed edge states allow the same model parameters to operate on different road graphs; runtime and state memory still depend on graph size.

Use this repository to study shared graph-recurrent architecture or adapt its graph and feature construction.

## Method

Represent roads as graph nodes and their spatial and temporal relationships as edges. Shared edge LSTMs encode those relationships; each node LSTM combines the incident-edge states with its own history to predict future speed.

### Algorithms and source

The [structural RNN reference](docs/structural-rnn.md) explains the equations, tensor contract, and implementation choices.

| Capability | Journal paper | Source or example |
|---|---|---|
| Graph and node/edge features | Sections III-A–B | [st_graph.py](st_graph.py), [dataset guide](dataset/Santander/README.md) |
| Shared spatial and temporal LSTMs | Eqs. (1)–(4) | `EdgeRNN` in [model.py](model.py) |
| Incident-edge aggregation and node prediction | Eqs. (5)–(11) | `SRNN.forward`, `NodeRNN` in [model.py](model.py), [synthetic example](example_synthetic.py) |
| Scaling, split, sequence batching | Section IV-A | [dataLoader.py](dataLoader.py), [MinMaxScaler.py](MinMaxScaler.py) |
| Training and cross-graph evaluation | Sections III-E, IV-B–C | [main_SRNN.py](main_SRNN.py), [running guide](docs/running.md) |

## Examples

Clone the repository:

```bash
git clone https://github.com/rhymesg/SRNN.git
```

Enter its root:

```bash
cd SRNN
```

Create a virtual environment with a Python version supported by [PyTorch](https://pytorch.org/get-started/locally/):

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux (Windows: `.venv\Scripts\activate`):

```bash
source .venv/bin/activate
```

Install the dependencies used by the source:

```bash
python -m pip install torch numpy pandas
```

See [verification status](docs/limitations.md#verification-status) for the checked modern CPU environment; original dependency versions were not recorded.

Run the synthetic forward example without using the Santander data or writing model files:

```bash
python example_synthetic.py
```

It builds a three-node directed chain and checks finite predictions with shape `(3, 3, 1)` from an untrained model. It demonstrates feature construction and the forward path, not prediction accuracy.

Run a small one-epoch training/evaluation check on the included dataset 1:

```bash
python main_SRNN.py --numData_set 128 --numData_train_set 96 --batch_size 2 --num_epochs 1
```

This writes or overwrites `save/dataset_1/srnn_model_epoch1.tar` and `log/_loss_eval_dataset_1.csv`. Run from the repository root; see the [running guide](docs/running.md) before changing the data size or epoch count.

## Implementation scope

The source includes graph construction, shared LSTM modules, training, and evaluation. Synthetic forward and training checks exercise those paths; [optimizer, data, and metric differences](docs/limitations.md) matter when comparing results with the papers.

### Checks

Run the [exact-batch and two-epoch training checks](tests/integration/training/README.md):

```bash
PYTHONPATH=. python tests/integration/training/verify_training.py
```

## Citation

Please cite the journal paper when using this method:

> Youngjoo Kim, Peng Wang, and Lyudmila Mihaylova. “Scalable Learning With a Structural Recurrent Neural Network for Short-Term Traffic Prediction.” *IEEE Sensors Journal*, 19(23), 11359–11366, 2019. [doi:10.1109/JSEN.2019.2933823](https://doi.org/10.1109/JSEN.2019.2933823). [Full text](https://arxiv.org/pdf/2103.02578v1); [ResearchGate](https://www.researchgate.net/publication/335076235_Scalable_Learning_with_a_Structural_Recurrent_Neural_Network_for_Short-Term_Traffic_Prediction).

The preliminary work describes the earlier architecture and experiments:

> Youngjoo Kim, Peng Wang, and Lyudmila Mihaylova. “Structural Recurrent Neural Network for Traffic Speed Prediction.” *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2019. [doi:10.1109/ICASSP.2019.8683670](https://doi.org/10.1109/ICASSP.2019.8683670). [Accepted manuscript](https://eprints.whiterose.ac.uk/id/eprint/142718/1/ICASSP_Kim_Wang_Mihaylova_2019.pdf); [ResearchGate](https://www.researchgate.net/publication/331222757_Structural_Recurrent_Neural_Network_for_Traffic_Speed_Prediction).

[CITATION.cff](CITATION.cff) contains software metadata, the preferred journal citation, and the preliminary paper reference. The paper citation request is separate from license obligations.

## License and provenance

The repository includes an [MIT license](LICENSE). See [provenance and limitations](docs/limitations.md) for implementation scope and the [dataset guide](dataset/Santander/README.md) for data-source and processing details.
