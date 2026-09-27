# Synthetic SRNN forward path; see docs/structural-rnn.md and README.md#citation.
# Method: https://doi.org/10.1109/JSEN.2019.2933823

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from main_SRNN import forward
from MinMaxScaler import MinMaxScaler
from model import SRNN
from st_graph import ST_GRAPH


def main():
    args = SimpleNamespace(
        batch_size=1, seq_length=3, node_input_size=1, edge_input_size=2,
        node_output_size=1, node_rnn_size=8, edge_rnn_size=8,
        node_embedding_size=4, edge_embedding_size=4, dropout=0.0,
    )
    torch.manual_seed(0)
    speeds = np.array([[30, 45, 60], [33, 42, 57],
                       [36, 39, 54], [39, 36, 51]], dtype=np.float32)
    scaler = MinMaxScaler()
    scaler.fit(speeds, feature_range=(0, 1), Min=0, Max=150)
    graph = ST_GRAPH(args)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "adjacency.csv"
        np.savetxt(str(path), [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
                   delimiter=",", fmt="%d")
        graph.readGraph(3, str(path))
    graph.putSequenceData(scaler.scale(speeds))
    features = graph.getSequenceData()
    net = SRNN(args)
    net.setStgraph(graph)
    net.eval()
    optimizer = torch.optim.Adagrad(net.parameters())
    with torch.no_grad():
        loss, _, outputs = forward(net, optimizer, args, graph, *features)
    assert tuple(outputs.shape) == (3, 3, 1)
    assert torch.isfinite(outputs).all().item() and torch.isfinite(loss).item()
    print("feature shapes:", [array.shape for array in features])
    print("prediction shape:", tuple(outputs.shape))
    print("Synthetic forward check passed (untrained model; no accuracy claim).")


if __name__ == "__main__":
    main()
