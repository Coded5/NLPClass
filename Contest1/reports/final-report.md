# Contest 1: Final Experiment Report

## Executive summary

This project predicts the exact next word from a text context and a literal
first-character hint. The work progressed from deterministic n-gram and GRU
baselines to a hint-restricted GPT-2 generator, collision-aware full-word
rerankers, multi-checkpoint fusion, and finally a neural candidate reranker.

The strongest system uses:

1. the step-258,000 GPT-2 checkpoint to generate the top 10 allowed BPE roots;
2. up to five complete training-corpus words for each root;
3. frozen 384-dimensional `all-MiniLM-L6-v2` embeddings for contexts and words;
4. grouped out-of-fold listwise MLP rerankers; and
5. calibrated bonuses for the earlier checkpoint-fusion and full-word-control
   predictions.

On 65,619 non-contaminated examples with alphanumeric hints, the final tuned
ensemble achieved:

| Metric | Result |
|---|---:|
| Top-1 accuracy | **63.3368%** (41,561/65,619) |
| Top-5 accuracy | **83.6236%** |
| Mean reciprocal rank | **0.72117** |
| Candidate oracle coverage | **88.6496%** |
| Grouped-bootstrap top-1 interval | **[62.9422%, 63.7325%]** |

This is a gain of **2.0360 percentage points** over the exactly recomputed
step-258 GPT-2 representative-word baseline and **0.8260 points** over the
checkpoint-fusion system on the same final population.

If all 7,903 non-alphanumeric examples are assumed correct, the corresponding
aggregate is 49,464/73,522 = **67.2778%**. This is an assumption-adjusted figure,
not a directly observed all-hint result.

The final number is also post-selection development evidence. Neural ensemble
weights and prediction bonuses were selected using these same grouped OOF
labels, so neither the point estimate nor its fixed-configuration bootstrap
interval is an unbiased holdout estimate.

## 1. Task and metrics

Each example contains:

- a tokenized text context;
- the literal first character of the next word; and
- for development evaluation, the complete target word.

Predictions must be complete words and must begin with the supplied character.
Case and punctuation are literal. The principal metric is exact-match top-1
accuracy. Top-5 accuracy, mean reciprocal rank (MRR), candidate coverage, and
paired promotion/breakage counts are used to diagnose ranking quality.

Candidate oracle coverage is the proportion of examples for which the exact
answer appears anywhere in the generated pool. It is an upper bound for every
reranker that uses that pool.

## 2. Data, populations, and evaluation policy

### 2.1 Canonical inputs

- `train/train.src.tok` is the punctuation-preserving corpus used by the final
  GPT-2 and reranker artifacts.
- The corpus contains 3,803,957 lines, 128,282,272 whitespace-delimited words,
  99,021 distinct words, and 155,102,098 GPT-2 BPE tokens.
- `data/devv_eval.csv` is the allowed development file and contains 75,860
  examples.
- `data/test_set_no_answer.csv` is the answer-free contest input.
- `data/train.src.tok` is a differently normalized corpus retained for the
  original notebook and is not interchangeable with the final corpus.

The canonical corpus SHA-256 is
`b46b640660bcfd2630dd18846edd9ac0f1769eff6caf3d5f70b46d40094a0bf9`.
The allowed development SHA-256 is
`1059c9b56995291a414adcf8d784ae941dd8ef6629cc17f2aaa1ba1a96be3420`.

### 2.2 Evaluation populations

Several reports use different, intentionally defined subsets. Their metrics
must not be compared without accounting for the scope.

| Population | Examples | Purpose |
|---|---:|---|
| Complete allowed development file | 75,860 | Full checkpoint diagnostics |
| Checkpoint-selection-contaminated rows | 2,338 | Excluded from later OOF comparisons |
| Non-contaminated development population | 73,522 | Checkpoint-fusion OOF evaluation |
| Non-alphanumeric hints in that population | 7,903 | Nearly deterministic route |
| Final non-contaminated alphanumeric population | 65,619 | Primary neural-reranker evaluation |
| Alphanumeric rows before contamination exclusion | 67,712 | Candidate preparation only |

The checkpoint selection subset contained 2,000 sampled rows. Because every
duplicate `(context, hint)` group was treated as one unit, excluding all groups
touched by selection removed 2,338 rows.

### 2.3 Leakage and contamination controls

- Identical `(context, hint)` groups are assigned to the same fold.
- Group folds are deterministic SHA-256-derived assignments.
- Candidate selection is label-independent.
- Scalar normalization is fitted only on each fold's training rows.
- Checkpoint-selection-contaminated groups are excluded from OOF comparisons.
- Paired uncertainty intervals resample complete groups, not individual rows.
- Candidate and prediction artifacts are checked for row alignment, finite
  values, hashes, duplicate candidates, and missing examples.
- `data/devv_test.csv` is forbidden. Evaluation entry points reject its name,
  canonical path, symlink target, and recorded content fingerprint before
  parsing it.

No result in this report was computed from the forbidden partition.

## 3. Implemented modeling routes

### 3.1 N-gram route

`n_gram_generator.py` implements a persistent SQLite n-gram word predictor with
literal first-character filtering and context backoff. `contest_pipeline.py`
adds CSV evaluation, submission generation, and strict output validation.

This route was implemented and covered by deterministic tiny-corpus tests,
including punctuation, numeric and uppercase hints, tie-breaking, effective
order reuse, CSV quoting, duplicate rows, backoff, and output order. No
production-scale n-gram accuracy result was retained in the current experiment
records, so this implementation is not assigned a numerical baseline here.

### 3.2 GRU neural language-model route

`neural_language_model.py` implements a first-character-conditioned GRU model
with a hint-restricted output vocabulary, streaming training examples,
checkpoint/resume support, exact CSV evaluation, and atomic prediction output.
Its checkpoint, restricted softmax, data-worker limits, and deterministic
behavior are tested. No comparable full-development GRU experiment result was
recorded, so it is documented as an implemented alternative rather than a
measured competitor.

### 3.3 Hint-masked GPT-2 route

The primary generator starts from `gpt2` and trains with supervision restricted
to roots allowed by the literal first-character hint. A root is the first GPT-2
BPE token used to begin one or more complete corpus words. The initial decoder
maps a selected root to its most frequent representative word.

The recorded training configuration was:

| Setting | Value |
|---|---:|
| Block size | 128 BPE tokens |
| Maximum evaluation context | 256 BPE tokens |
| Micro-batch size | 8 |
| Gradient accumulation | 4 |
| Effective batch size | 32 |
| Learning rate | 5e-5 |
| Weight decay | 0.01 |
| Warmup | 200 updates |
| Gradient clipping | 1.0 |
| Planned cosine schedule | 8 epochs / 302,928 updates |
| Updates per epoch | 37,866 |
| Evaluation cadence | every 2,000 updates on 2,000 examples |
| Precision | FP16 on CUDA |
| Seed | 42 |

Training was resumable and atomically checkpointed. The preserved checkpoints
used in the final analyses are step 126,000, step 258,000, and step 276,161.
The planned schedule was not completed; a remote shutdown snapshot preserved
the latest state.

## 4. GPT-2 checkpoint experiments

### 4.1 Comparable full-development checkpoint evaluation

The comprehensive cache-first evaluator scored all 75,860 allowed development
examples with probabilities normalized over every root permitted by the hint.
These are restricted-root probabilities, not probabilities over the complete
GPT-2 vocabulary.

| Checkpoint | Top-1 | Top-5 | Mean restricted entropy |
|---|---:|---:|---:|
| Step 126,000 | 64.9987% | 81.9457% | 0.9811 |
| Step 258,000 | 65.4390% | **82.1632%** | 0.9877 |
| Step 276,161 | **65.5075%** | 82.0643% | 0.9103 |

Additional training improved top-1 only modestly. Step 276,161 was the best
single checkpoint for top-1, while step 258,000 had the best top-5 result and
was selected as the final candidate generator because reranking benefits from
candidate recall rather than only root top-1.

Standalone step-126/step-258 reports produced 64.9974% and 65.4403% top-1,
respectively. Those artifacts differ from the comprehensive cache by one
prediction at each checkpoint. For cross-checkpoint comparisons, this report
uses the comprehensive evaluator above; for the final filtered population, it
uses the exact recomputation from the final candidate artifacts.

### 4.2 Checkpoint interpolation

A probability-space sweep blended steps 126,000 and 258,000:

| Step-126 weight | Step-258 weight | Top-1 | Top-5 | Net top-1 vs step 258 |
|---:|---:|---:|---:|---:|
| 0.0 | 1.0 | 65.4390% | 82.1632% | 0 |
| 0.1 | 0.9 | 65.5840% | 82.2107% | +110 |
| 0.2 | 0.8 | 65.6538% | 82.2476% | +163 |
| **0.3** | **0.7** | **65.7132%** | 82.2792% | **+208** |
| 0.4 | 0.6 | 65.6710% | 82.3135% | +176 |
| 0.5 | 0.5 | 65.6472% | 82.2963% | +158 |

Weights above 0.7 on step 126 became harmful. An equal unnormalized sum of all
three checkpoints also reached 65.7132% top-1 and 82.3253% top-5. These sweeps
used the complete development labels and are descriptive tuning results, not
selection-safe estimates.

### 4.3 Error analysis

The step-258 diagnostics identified several distinct limits:

- 12,682 top-1 errors had the exact representative word at ranks 2 through 5.
- 4,915 targets shared a BPE root with a more frequent representative word.
  A root-only decoder cannot emit those answers; in 2,814 direct-evaluation
  cases the correct root was already ranked first.
- Targets seen at most 100 times or absent from training accounted for 868
  examples and achieved only 1.73% top-1 accuracy in the comprehensive report.
- Candidate distributions and target diversity varied substantially by hint.
  `c` had the highest empirical target entropy at 5.81 nats and 52.38% top-1.
- Among well-represented letters, `o` was strongest at 82.69% top-1, while `s`
  was weakest at 39.47%.

Performance fell sharply with answer BPE length:

| Answer BPE length | Examples | Top-1 | Top-5 |
|---:|---:|---:|---:|
| 1 | 68,022 | 69.98% | 87.60% |
| 2 | 5,581 | 26.66% | 37.68% |
| 3 | 1,811 | 17.78% | 22.36% |
| 4 | 422 | 55.21% | 55.45% |
| 5+ | 24 | 0.00% | 0.00% |

Frequency was similarly decisive:

| Training frequency | Examples | Top-1 | Top-5 |
|---:|---:|---:|---:|
| 2-5 | 26 | 0.00% | 0.00% |
| 6-10 | 70 | 0.00% | 0.00% |
| 11-100 | 772 | 1.94% | 4.40% |
| 101-1,000 | 3,301 | 18.84% | 34.08% |
| 1,001+ | 71,691 | 68.36% | 85.32% |

The model was often overconfident when wrong. Wrong top-1 predictions had
52.83% mean and 49.62% median restricted confidence. Of the errors, 49.50% had
at least 50% confidence, 11.96% had at least 90%, and 4.24% had at least 99%.
Of 1,112 errors at 99% confidence or higher, 930 were correct-root rank-1
collisions. This motivated complete-word reranking rather than confidence
calibration alone.

## 5. Complete-word reranking experiments

### 5.1 First leakage-safe top-five reranker

The comprehensive analysis built a supervised reranker over the top-five root
representatives using model score/rank, top-1 margin, unigram, hint-conditioned
unigram, bigram and trigram counts, character length, BPE length, and context
length. N-gram features were counted only from training text.

The development data was grouped into 43,984 training, 14,624 validation,
14,914 internal test, and 2,338 contaminated rows. Here, “internal test” means
an untouched split of `devv_eval.csv`; it is not the forbidden file.

| Split | Root baseline | Reranker | Gain | Promotions | Breakages |
|---|---:|---:|---:|---:|---:|
| Validation | 65.2557% | 65.8780% | +0.6223 pp | 198 | 107 |
| Internal test | 65.8039% | **66.5549%** | **+0.7510 pp** | 234 | 122 |

The positive held-out development-split result established that training-only
lexical features could recover ranking errors without excessive breakage.

### 5.2 Collision-aware heuristic and learned-ranker study

A wider step-126 candidate cache used five roots and up to five complete words
per root, with suffix and boundary scores. It was divided into 11,019 tuning
training rows, 3,690 tuning-validation rows, 58,813 locked rows, and 2,338
contaminated rows.

On the 3,690-row tuning-validation split:

| Method | Top-1 | Gain over 65.7724% baseline | Promotions | Breakages |
|---|---:|---:|---:|---:|
| Best hand-tuned heuristic | 65.8266% | +0.0542 pp | 60 | 58 |
| Cheap learned features | 65.8537% | +0.0813 pp | 7 | 4 |
| Learned features without boundary score | 66.1247% | +0.3523 pp | 19 | 6 |
| Full learned ranker | **66.2602%** | **+0.4878 pp** | 27 | 9 |

The best heuristic used five words per root, suffix weight 0.1, length penalty
1.0, frequency weight 0.05, boundary weight 0.25, and confidence gate 0.5. Its
small net gain showed that fixed scoring rules were too brittle.

The full learned ranker used root probability/rank/margin/entropy, within-root
rank, suffix probability and length, boundary probability, frequency, root
candidate count, representative status, and numeric/alphabetic indicators.
With confidence gate 0.5, it generalized to the 58,813 locked rows:

- baseline: 65.0434%;
- ranker: **65.6777%**;
- gain: **+0.6342 points** with 95% CI `[+0.5434, +0.7346]`;
- 494 wrong-to-correct promotions and 121 correct-to-wrong breakages.

### 5.3 Multi-checkpoint fusion

Checkpoint fusion retained the full-word candidate pool and histogram-gradient-
boosting control, then added step-258 and step-276,161 root log-probabilities.
Both control and treatment used five grouped OOF folds, learning rate 0.05, 200
iterations, 31 maximum leaf nodes, L2 regularization 1.0, seed 42, and a fixed
0.5 confidence gate.

On all 73,522 non-contaminated examples:

| System | Top-1 |
|---|---:|
| Step-126 root-only | 64.9588% |
| Step-258 root-only | 65.3981% |
| Step-276,161 root-only | 65.4756% |
| Full-word control | 65.9762% |
| **Fusion treatment** | **66.4753%** |

Fusion improved on the full-word control by **0.4992 points**, with grouped
95% CI `[+0.3852, +0.6105]`, 1,078 promotions, and 711 breakages. It improved
on the best single root checkpoint by 0.9997 points.

On the final 65,619-example alphanumeric subset:

| System | Top-1 |
|---|---:|
| Step-126 root-only | 60.8117% |
| Step-258 root-only | 61.3039% |
| Step-276,161 root-only | 61.3908% |
| Full-word control | 61.9516% |
| **Fusion treatment** | **62.5109%** |

The alphanumeric fusion gain over control was **0.5593 points**, with 95% CI
`[+0.4265, +0.6945]`. The gain over step 276,161 was 1.1201 points, with 95% CI
`[+0.9354, +1.3121]`.

Fusion ranking metrics over all non-contaminated hints were:

| Metric | Control | Fusion |
|---|---:|---:|
| Candidate oracle coverage | 85.1908% | 85.1908% |
| Top-5 accuracy | 84.2415% | 84.3285% |
| MRR | 0.73545 | 0.74105 |

The benefit was concentrated in the 11,397 examples where checkpoints
disagreed: fusion improved over control by 3.1850 points there, versus only
0.0064 points on 62,125 agreement examples. The candidate pool remained a
major limit: 9,646 targets were outside the five-root beam and 1,242 were beyond
the five-word per-root limit.

The estimated scoring cost was 59.2 seconds for root-only, 572.2 seconds for
full-word scoring, and 690.6 seconds for fusion over the complete development
set. Fusion was approximately 11.7 times root-only cost and 1.2 times the
full-word pipeline cost, excluding model loading, score merging, and classifier
inference.

## 6. Neural candidate reranker

### 6.1 Motivation and design

The error studies showed that the true word was frequently available but
misranked, while fixed heuristics and tree models could not fully exploit
semantic compatibility between a context and a complete candidate word.

The final neural design is deliberately small:

- frozen `sentence-transformers/all-MiniLM-L6-v2` encoder;
- masked mean pooling followed by L2 normalization;
- float16 cached embeddings, 384 dimensions each;
- separate context and candidate-word embeddings;
- context, candidate, elementwise product, and absolute-difference interaction
  vectors;
- 12 normalized scalar GPT-2 and lexical features;
- a listwise MLP with GELU activations and 0.1 dropout;
- cross-entropy over all candidates belonging to one example; and
- deterministic tie-breaking by score, root rank, within-root rank, and word.

The scalar features are root log-probability, root rank, within-root rank, root
margin, restricted entropy, log word count, root candidate count, BPE length,
character length, numeric status, alphabetic status, and representative status.

A width-256 MLP has 429,569 trainable parameters; width 512 has 924,673. MiniLM
itself has 22,713,216 frozen parameters, compared with 124,439,808 in GPT-2.

### 6.2 Candidate preparation

The selected generator was step 258,000 with a beam of 10 roots and up to five
complete words per root. Candidate generation remained independent of the
answer labels.

Preparation produced:

| Artifact statistic | Value |
|---|---:|
| Alphanumeric examples before contamination exclusion | 67,712 |
| Candidate rows | 2,027,803 |
| Unique contexts | 64,444 |
| Unique candidate words | 28,864 |
| Embedding dimension | 384 |
| Maximum embedding length | 256 tokens |
| Prepared artifact size | 121 MiB |

After excluding contaminated groups, 65,619 examples and 1,965,092 candidate
rows remained. The pool's oracle coverage was 88.6496%:

| Candidate status | Examples |
|---|---:|
| Exact answer reachable | 58,171 |
| Correct root outside top-10 beam | 6,044 |
| Correct word outside five-word limit | 1,404 |

The wider beam raised oracle coverage above the earlier 85.19% full-word pool,
but 11.35% of the final examples still could not be solved by any reranker.

### 6.3 Grouped OOF neural ablations

Every configuration used five deterministic grouped OOF folds, batch size 128,
learning rate 1e-3, dropout 0.1, and fold-only scalar normalization.

| Hidden width | Epochs | Seed | Top-1 | Top-5 | MRR | Command time |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 5 | 42 | 62.5596% | **83.5505%** | 0.71634 | 96.48 s |
| 256 | 10 | 42 | 62.6038% | 83.4408% | 0.71622 | 169.97 s |
| 512 | 5 | 42 | 62.5916% | 83.4804% | 0.71647 | 107.89 s |
| 512 | 10 | 42 | 62.6450% | 83.4179% | 0.71643 | 192.55 s |
| 512 | 10 | 43 | **62.7014%** | 83.3829% | **0.71668** | 192.96 s |

All individual models improved clearly over the exactly recomputed step-258
root baseline of 61.3008%. The first width-256/five-epoch model gained 1.2557
points over the imported step-258 baseline, 0.6081 over the full-word control,
and 0.0488 over checkpoint fusion. Its fusion comparison interval included
zero. The seed-43 width-512 model was the strongest individual run at 62.7014%,
but model diversity proved more useful than selecting only that model.

Training losses decreased throughout the longer runs. The small top-1 changes,
mixed top-5 changes, and gains from blending show that the variants learned
different rankings rather than forming a simple monotonic capacity curve.

## 7. Neural blending and the final ensemble

### 7.1 Blend progression

| Experiment | Top-1 | Interpretation |
|---|---:|---|
| Best individual neural model | 62.7014% | Width 512, 10 epochs, seed 43 |
| Exploratory 50/50 two-run blend | 62.8812% | Capacity/epoch diversity helped |
| Equal blend of three then-available runs | 62.8355% | Weaker than tuned two-run blend |
| Tuned 45/55 two-run neural blend | 62.8934% | Development-selected weight |
| Tuned blend plus fusion/control bonuses | 63.2286% | Earlier two-neural final candidate |
| Width-512 seed-42/seed-43 blend | 62.8461% | Seed diversity alone |
| **Final hierarchical ensemble** | **63.3368%** | Three neural models plus priors |

The width-512 seed ensemble used 35.8% seed 42 and 64.2% seed 43 and achieved
83.5017% top-5 and 0.71790 MRR. The final ensemble then assigned 33% to the
width-256/five-epoch model and 67% to that seed ensemble. Its effective neural
weights are therefore:

- width 256, five epochs, seed 42: **0.33000**;
- width 512, ten epochs, seed 42: **0.23986** (reported as 0.24); and
- width 512, ten epochs, seed 43: **0.43014** (reported as 0.43).

After combining neural logits, the final score adds:

- **0.23** when a candidate matches the checkpoint-fusion prediction; and
- **0.375** when it matches the earlier full-word-control prediction.

These bonuses act as calibrated prediction priors. They preserve complementary
signals from the tree models without requiring them to share a score scale with
the neural rerankers.

### 7.2 Final paired comparisons

| Baseline | Baseline top-1 | Final gain | Promotions | Breakages | 95% grouped CI |
|---|---:|---:|---:|---:|---:|
| Exact step-258 root reconstruction | 61.3008% | **+2.0360 pp** | 2,536 | 1,200 | `[+1.8309, +2.2378]` |
| Width-256/five-epoch neural | 62.5596% | +0.7772 pp | 1,576 | 1,066 | `[+0.6063, +0.9344]` |
| Width-512 seed ensemble | 62.8461% | +0.4907 pp | 1,598 | 1,276 | `[+0.3216, +0.6525]` |
| Checkpoint fusion | 62.5109% | **+0.8260 pp** | 1,689 | 1,147 | `[+0.6645, +0.9989]` |
| Full-word control | 61.9516% | **+1.3853 pp** | 2,275 | 1,366 | `[+1.2025, +1.5670]` |

The imported checkpoint-fusion table reports step-258 at 61.3039%, two correct
examples above the exact final candidate reconstruction. Against that imported
version the gain is 2.0329 points. The headline 2.0360-point gain uses the
baseline recomputed directly from the exact final candidate population:
40,225 correct versus 41,561 for the final ensemble.

## 8. Runtime and resource measurements

The main preparation and training measurements were:

| Operation | Time / resource |
|---|---:|
| Candidate-pool construction | 21.98 s |
| Initial MiniLM embedding | 11.92 s |
| Complete preparation command | 35.32 s recorded / 38.98 s wall |
| Default five-fold width-256 training | 74.31 s |
| Default OOF inference | 2.91 s |
| Independent seed-43 training | 170.56 s |
| Independent seed-43 OOF inference | 3.16 s |
| Final offline ensemble command | 8.18 s |
| Final ensemble peak process RSS | 9,130 MiB |

A separate embedding benchmark re-encoded all 64,444 contexts and 28,864 words
in 11.94 seconds. It used 1,579.9 MiB peak process RSS, 277.3 MiB peak CUDA
allocation, and 740.0 MiB peak CUDA reservation. The seed-43 training run used
4,288.6 MiB peak process RSS.

The experiments ran on an NVIDIA GeForce RTX 3070 with 8 GB VRAM. Before the
neural runs, the host had 31 GiB RAM, 22 GiB available memory, 27 GiB swap, and
107 GiB free disk.

## 9. What worked and what did not

### Effective changes

1. **Training GPT-2 longer helped, but only modestly.** Step 258 improved on
   step 126 by roughly 0.44 top-1 points; step 276 added only about 0.07 more.
2. **Checkpoint diversity was real.** Simple step-126/step-258 interpolation
   added 0.2742 points on the full development file, and learned fusion added
   0.5593 points over the full-word control on alphanumeric OOF rows.
3. **Complete-word candidates addressed root collisions.** The learned
   full-word ranker clearly outperformed the root representative baseline.
4. **Learned reranking beat fixed heuristics.** The best heuristic gained only
   0.0542 points on tuning validation, while the full learned ranker gained
   0.4878 and then 0.6342 on its locked split.
5. **Semantic context/candidate features added another step.** Every MiniLM MLP
   exceeded the exact step-258 root baseline by at least 1.25 points.
6. **Diversity beat a single best model.** Blending different capacities,
   training durations, and random seeds was stronger than any constituent.
7. **Earlier model predictions remained complementary.** Fusion/control bonuses
   raised the tuned neural system from the low 62.9% range to 63.3368%.

### Limited or unsuccessful changes

1. Hand-tuned suffix, frequency, length, boundary, and confidence-gating rules
   produced many offsetting promotions and breakages.
2. Increasing width or epochs alone yielded only small and non-monotonic gains.
3. Later GPT-2 checkpoints did not improve all ranking metrics: step 276 had
   slightly better top-1 but worse top-5 than step 258.
4. Candidate reranking cannot solve missing candidates. Even the final pool
   excludes 7,448 exact answers, fixing the oracle ceiling at 88.6496%.
5. Rare and multi-BPE-token words remained difficult despite reranking.

## 10. Statistical interpretation and limitations

The strongest evidence before final ensemble tuning came from grouped OOF or
internally locked development comparisons. Those experiments demonstrated real
paired improvements and prevented duplicate context/hint groups from crossing
folds.

However, the complete sequence involved repeated development-driven choices:
checkpoint selection, candidate width, MLP width, epoch count, random-seed
blending, neural weights, and prediction bonuses. The final 63.3368% result is
therefore a post-selection development result. Its bootstrap interval describes
the saved predictions under the selected configuration; it does not account for
the uncertainty introduced by choosing that configuration on the same labels.

There is no recorded final accuracy on an untouched answer-bearing test set,
and the forbidden partition was not used. There is also no directly observed
final all-hint score: 67.2778% assumes perfect handling of the 7,903
non-alphanumeric examples, while earlier systems measured 99.3926% on that
route rather than exactly 100%.

The correct deployment-quality next step would be nested grouped cross-
validation or a genuinely untouched holdout evaluated once after freezing all
weights and bonuses.

## 11. Reproducibility and verification

The final implementation records:

- hashes for the canonical corpus, development file, checkpoints, GPT-2 score
  caches, candidate files, MiniLM files, prepared embeddings, fold models, and
  final outputs;
- exact package, CUDA, cuDNN, and GPU versions;
- fold-specific training counts, normalization statistics, losses, timings,
  and model hashes;
- deterministic seeds and ranking tie-breakers;
- complete OOF candidate scores and predictions; and
- artifact manifests for the source and ensemble runs.

The final saved artifacts were independently checked for all 1,965,092
candidate scores, prediction agreement, duplicate candidates, missing rows,
fold crossings, and contamination. The final score was reproduced as exactly
41,561/65,619. All five final artifact hashes and 40 recorded hashes across the
three source-model and two ensemble runs verified.

After project reorganization:

- all 59 tests passed;
- all nine installed command-line tools completed `--help` outside the checkout;
- source and wheel distributions built successfully;
- `uv pip check` found all 152 installed packages compatible;
- six permitted corpus/data hashes matched the manifest, while the forbidden
  partition was deliberately skipped; and
- an independent review found no remaining blocker or major issue.

## 12. Artifact map

Generated artifacts are intentionally excluded from Git. The main local paths
are:

| Purpose | Path |
|---|---|
| Checkpoint caches and comprehensive analysis | `artifacts/comprehensive-evaluation/` |
| GPT-2 checkpoints, tokenizer, logs, and direct evaluations | `artifacts/gpt2-keyboard-script/` |
| Step-126 complete-word candidate cache | `artifacts/gpt2-keyboard-script/rerank/` |
| Full-word locked result | `artifacts/gpt2-keyboard-script/rerank/final/` |
| Checkpoint-fusion OOF result | `artifacts/gpt2-keyboard-script/rerank/checkpoint-fusion/` |
| Frozen neural preparation | `artifacts/gpt2-keyboard-script/rerank/neural-shared-context/` |
| Width-256/five-epoch OOF run | `artifacts/gpt2-keyboard-script/rerank/neural-shared-context-oof/` |
| Width-256/ten-epoch OOF run | `artifacts/gpt2-keyboard-script/rerank/neural-shared-context-oof-e10/` |
| Width-512/five-epoch OOF run | `artifacts/gpt2-keyboard-script/rerank/neural-shared-context-oof-h512-e5/` |
| Width-512/ten-epoch seed-42 run | `artifacts/gpt2-keyboard-script/rerank/neural-shared-context-oof-h512-e10/` |
| Width-512/ten-epoch seed-43 run | `artifacts/gpt2-keyboard-script/rerank/neural-shared-context-oof-h512-e10-seed43/` |
| Width-512 seed ensemble | `artifacts/gpt2-keyboard-script/rerank/neural-shared-context-h512-seed-ensemble/` |
| Final ensemble | `artifacts/gpt2-keyboard-script/rerank/neural-shared-context-ensemble/` |

Tracked documentation is split by purpose:

- `README.md` gives the short project overview;
- `reports/neural-reranker.md` gives the compact final result;
- this report gives the complete methodology and experiment history; and
- `LOGS.md` preserves the chronological execution record.

## 13. Project reorganization and recovery note

The work was packaged under `src/contest1`, tests were moved to `tests`,
notebooks to `notebooks`, generated state to `artifacts`, and local data to the
documented `data` and `train` locations. Console entry points, project-root path
handling, dependency metadata, data manifests, and fail-closed evaluation
guards were added.

During that reorganization, a text-oriented move corrupted two untracked binary
source files: the assignment PDF and a redundant original training ZIP. No clean
copy was found locally or in related repository history, and attempted PDF
recovery produced blank pages. The damaged files and recovery attempt are kept
under ignored `artifacts/recovery/` and are not part of the proposed commit.
This incident did not affect the canonical training corpus, allowed development
data, checkpoints, candidate caches, predictions, metrics, or final model
artifacts.

## Conclusion

The main modeling lesson is that this task is not solved by root prediction
alone. GPT-2 learned strong hint-conditioned root distributions, but complete-
word collisions, multi-token words, and candidate ordering left substantial
recoverable error. Widening to complete-word candidates, combining checkpoints,
and learning semantic context/candidate interactions produced consistent gains.

The final tuned system reaches **63.3368% top-1**, **83.6236% top-5**, and
**0.72117 MRR** on the carefully filtered alphanumeric development population.
It is the strongest saved system in the project, with reproducible artifacts
and validated computations. Its remaining limitation is evaluation design, not
an identified implementation defect: the final ensemble still requires one
truly untouched, configuration-frozen evaluation before its accuracy can be
treated as a generalization estimate.
