```mermaid
graph TD
    A[Input: batch, 10] --> B[Compute Abs Values]
    B --> C[Concat: Original + Abs]
    C --> D[Expand Dims: batch, 10, 2]
    D --> E[Conv1D: 64 filters]
    E --> F[Conv1D: 32 filters]
    F --> G[GlobalAvgPool: batch, 32]
    G --> H[Dense: 10 logits]
    H --> I[MinFinderLossLayer]
    I --> J[Output: Scores]