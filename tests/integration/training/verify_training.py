from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
import torch
import main_SRNN as task

args = SimpleNamespace(batch_size=2, seq_length=3, numNodes_set=3,
    numData_set=16, numData_train_set=8, num_epochs=2, grad_clip=1.,
    node_input_size=1, edge_input_size=2, node_output_size=1,
    node_rnn_size=4, edge_rnn_size=4, node_embedding_size=4,
    edge_embedding_size=4, dropout=0.5, printEvery=100)
observed = []
original_forward = task.forward
def probe(net, *rest):
    observed.append((net.training, torch.is_grad_enabled()))
    return original_forward(net, *rest)
task.forward = probe
torch.manual_seed(0)
with TemporaryDirectory() as directory:
    root = Path(directory)
    data = root / 'data'; data.mkdir()
    speeds = np.arange(48, dtype=np.float32).reshape(16,3) % 50 + 30
    np.savetxt(data / 'Data_1.csv', speeds, delimiter=',')
    np.savetxt(data / 'Adjacency_1.csv', np.array([[0,1,0],[0,0,1],[0,0,0]]), delimiter=',', fmt='%d')
    task.data_dir = str(data) + '/'
    task.save_dir = str(root / 'save') + '/'
    task.log_dir = str(root / 'log') + '/'
    task.Run_SRNN_NormalCase(args, 1)
    assert observed == [(True,True)]*2 + [(False,False)]*2 + [(True,True)]*2 + [(False,False)]*2
    assert len(list((root/'save'/'dataset_1').glob('*.tar'))) == 2
    rows = np.loadtxt(root/'log'/'_loss_eval_dataset_1.csv',delimiter=',')
    assert rows.shape == (3,2) and np.isfinite(rows).all()
    assert list(rows[:2,0]) == [1,2]
    print('Exact-batch, two-epoch, and evaluation-mode checks passed.')
