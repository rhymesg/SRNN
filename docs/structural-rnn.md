# Structural recurrent neural network for traffic speed prediction

Implementation reference for [SRNN](https://github.com/rhymesg/SRNN), mapping the [journal paper](../README.md#citation), Section III, to the supplied PyTorch code. The preliminary ICASSP paper describes an earlier version; this reference follows the journal architecture.

## Problem and features

Given `L` observations on `N` road segments and a directed adjacency matrix, predict each segment's next speed. All nodes share one node-RNN parameter set; spatial edges share another, and temporal edges share a third.

For scaled speed `x[t,u]`, [ST_GRAPH.putSequenceData](../st_graph.py) constructs:

| Feature | Definition | Array shape from `getSequenceData()` |
|---|---|---|
| Node | `x[t,u]` | `(L+1, N, 1)` |
| Temporal edge | `[x[t-1,u], x[t,u]]` | `(L+1, N, 2)` |
| Spatial edge `(u,v)` | `[x[t,u], x[t,v]]` | `(L+1, E, 2)` |

At the sequence boundary, the temporal feature is `[x[0,u], x[0,u]]`. The graph reader creates a spatial edge for each matrix entry exactly equal to `1` and a temporal self-edge for each node; see [input restrictions](../dataset/Santander/README.md#input-contract).

## Forward path and equations

Let `phi` mean a learned affine embedding, ReLU, then dropout; let `C(u)` contain all spatial edges incident to `u`, including incoming and outgoing edges. At each time step:

1. Embed spatial and temporal features separately and update their LSTMs: `(hS,cS) = LSTM_S(phiS(xS), (hS,cS))`, and likewise for `(hT,cT)`.
2. Sum incident spatial states: `s[u] = sum(hS[e] for e in C(u))`.
3. Concatenate temporal and spatial context: `H[u] = concat(hT[u], s[u])`.
4. Update the node LSTM from `concat(phi_node(x[u]), phi_context(H[u]))`.
5. Apply the output affine layer to the node hidden state to obtain a scaled speed prediction.

| Journal equation | Implementation in [model.py](../model.py) |
|---|---|
| (1)–(4): edge embeddings and LSTMs | `EdgeRNN.forward`, separate `EdgeRNN_spatial` and `EdgeRNN_temporal` instances |
| (5)–(6): select and sum incident spatial states | `SRNN.forward`, `index_select` followed by `torch.sum` |
| (7): concatenate temporal and spatial states | `SRNN.forward`, `h_edgeRNN_eachNode` |
| (8)–(9): node and context embeddings | `NodeRNN.forward`, `encoder_linear` and `edge_embed` |
| (10): node recurrent update | `NodeRNN.cell`, an `nn.LSTMCell` |
| (11): predicted node feature | `NodeRNN.output_linear` |

The code uses a **sum**, not a degree-normalized average; the journal's Eq. (6) also specifies a sum despite describing an “average spatial influence.” Its Eq. (11) omits an explicit bias, while `nn.Linear` in this implementation includes one.

Fixed embedding and hidden-state widths keep the trainable parameter count independent of `N` and `E`. The arrays of states and the work to process them still grow with the graph; weights are shared, but recurrent states remain separate for each node/edge.

## Callable interface

- Construct `SRNN(args)` with `seq_length`, node/edge input sizes, node output size, node/edge hidden sizes, node/edge embedding sizes, and `dropout`.
- Call `setStgraph(graph)` before `forward`; otherwise its assertion fails.
- Pass float CPU tensors for nodes, temporal edges, and spatial edges covering `L` steps, then six hidden/cell tensors in node, temporal-edge, spatial-edge order.
- Hidden/cell shapes are `(N, node_rnn_size)`, `(N, edge_rnn_size)`, and `(E, edge_rnn_size)` for the three RNN types.
- The return value is predictions of shape `(L, N, node_output_size)`; final hidden/cell states are not returned.
- [main_SRNN.forward](../main_SRNN.py) accepts the `L+1` feature arrays, creates zero states per sequence, uses the first `L` steps as inputs, and computes MSE only between the final output and the held-out final observation.
- Outputs are unconstrained scaled values; [MinMaxScaler.scale_inverse](../MinMaxScaler.py) converts them back to speed units without clipping.

## Graph and execution model

The shared architecture treats road segments as semantically equivalent and uses a fixed supplied graph with CPU tensors.

The journal uses directed links; the ICASSP version introduced two opposing edges per connection. `readGraph` follows the supplied matrix and does not add reverse links automatically.

Run the [synthetic example](../example_synthetic.py) via the [README command](../README.md#examples) to inspect the feature and output shapes. Consult [implementation details](implementation-notes.md) before using training logs as research evidence.
