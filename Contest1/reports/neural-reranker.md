# Neural candidate reranker

## Evaluation scope

- 65,619 alphanumeric-hint examples
- All checkpoint-selection-contaminated groups excluded
- Identical `(context, hint)` groups assigned to one OOF fold
- Development OOF evidence only; ensemble parameters were selected on these
  same labels

## Performance

| System | Top-1 | Top-5 | MRR |
|---|---:|---:|---:|
| Step-258 GPT-2 representative-word ranking | 61.3008% | 80.0622% | 0.69787 |
| Best individual MiniLM reranker | 62.7014% | 83.3829% | 0.71668 |
| Final tuned ensemble | **63.3368%** | **83.6236%** | **0.72117** |

The final system improves top-1 by 2.0360 percentage points over the original
step-258 GPT-2 ranking and by 0.8260 points over checkpoint fusion. Candidate
oracle coverage is 88.6496%.

Assuming perfect accuracy on all 7,903 non-alphanumeric hints gives
49,464/73,522 = **67.2778%** aggregate accuracy.

## Final blend

- Width-256/five-epoch neural logits: 0.33
- Width-512/ten-epoch seed-42 neural logits: 0.24
- Width-512/ten-epoch seed-43 neural logits: 0.43
- Checkpoint-fusion prediction bonus: 0.23
- Full-word-control prediction bonus: 0.375

The complete generated outputs remain under
`artifacts/gpt2-keyboard-script/rerank/neural-shared-context-ensemble/` and are
excluded from Git. `LOGS.md` contains the chronological implementation,
training, benchmark, and verification record.
