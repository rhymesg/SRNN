# Santander traffic data

Input reference for the four bundled road subsets used by [SRNN](../../README.md). The [journal paper](../../README.md#citation), Section IV-A and Fig. 4, describes Santander traffic speed measurements in km/h aggregated at 15-minute intervals.

## Supplied files

All CSV files are numeric and have no header; data columns and adjacency rows/columns must describe road segments in the same order.

| Subset | Speed measurements | Directed adjacency | Auxiliary IDs | Rows × nodes | Spatial edges |
|---|---|---|---|---|---|
| 1 | [Data_1.csv](Data_1.csv) | [Adjacency_1.csv](Adjacency_1.csv) | [ID_1.csv](ID_1.csv) | 33,504 × 5 | 5 |
| 2 | [Data_2.csv](Data_2.csv) | [Adjacency_2.csv](Adjacency_2.csv) | [ID_2.csv](ID_2.csv) | 33,504 × 5 | 4 |
| 3 | [Data_3.csv](Data_3.csv) | [Adjacency_3.csv](Adjacency_3.csv) | [ID_3.csv](ID_3.csv) | 33,504 × 9 | 9 |
| 4 | [Data_4.csv](Data_4.csv) | [Adjacency_4.csv](Adjacency_4.csv) | [ID_4.csv](ID_4.csv) | 33,504 × 19 | 18 |

- `Data_k.csv`: rows are successive observations, columns are road segments; no timestamp column is supplied.
- `Adjacency_k.csv`: square binary matrix; entry `(u,v)=1` creates the ordered spatial feature `[speed_u, speed_v]`.
- `ID_k.csv`: three rows of identifiers per data column; the scripts never read these files, and the meaning of each ID row is not documented in the source.
- Every bundled adjacency matrix has a zero diagonal and no isolated nodes; dataset 4 comprises disconnected components, which is different from having isolated nodes.

## Input contract

- Provide finite numeric speeds and a matching square adjacency matrix; the loader does not impute or reject missing/non-finite speeds explicitly.
- Use `0` or `1` adjacency values: only exact `1` entries create spatial edges, so weighted adjacency is not supported.
- Keep the diagonal zero: temporal self-edges are added internally, and diagonal spatial edges conflict with them.
- Every node must have at least one incident spatial edge; the current `SRNN.forward` isolated-node branch has incompatible context dimensions.
- Supply enough observations for nonempty train and evaluation batches and the stricter [batch-boundary assertion](../../docs/running.md#sequence-boundaries-and-data-requirements).
- Use compatible speed units: [DataLoader](../../dataLoader.py) applies fixed bounds `0` and `150`, without clipping or deriving a min/max from the data.

## Preprocessing and provenance

The paper reports SETA project measurements from 2016 and describes filling missing values using other days' values at the same time. The supplied loader performs no such imputation, and the repository contains no raw-data conversion script or timestamp map.

The journal describes 35,040 readings per segment, whereas each bundled data file has 33,504 rows. Do not assume exact calendar coverage or a first-nine-months split from these files; the default loader uses a row-count split of 25,128 training and 8,376 evaluation observations.

The [MIT license](../../LICENSE) covers the repository's software. Confirm dataset redistribution terms with the data rights holder; the [synthetic example](../../example_synthetic.py) generates its own inputs.
