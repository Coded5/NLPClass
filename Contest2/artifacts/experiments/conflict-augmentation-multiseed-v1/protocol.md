# Multi-seed conflict augmentation isolation

This follow-up compares two RoBERTa polarity training sets over five grouped
outer folds and seeds 42, 1337, and 2024. Both add exactly 40 conflict-labelled
rows per fold with identical aspect counts. The repetition arm duplicates
specific real same-aspect conflict rows frozen in `preparation.json`; the
synthetic arm adds the corresponding reviewed same-aspect clause combinations.
Cross-aspect positive/negative controls from the first pilot are excluded.

Each pair uses the same model seed, natural fitting/selection/heldout IDs,
optimizer settings, epoch budget, frozen original-data class weights, and
checkpoint-selection metric. Repeated-row identities stay fixed across seeds,
so seed variation covers initialization and training order rather than changing
the control data. Neither historical validation nor historical test is read.

The primary comparison averages logits across the three seeds within every
condition and scores the resulting natural out-of-fold predictions. Report
conflict F1, all class metrics, accuracy, macro-F1, conflict false-positive
rate, per-seed pooled results, all 15 fold-seed deltas, timing, selected epochs,
and a paired review-bootstrap 95% interval with 10,000 samples.

Synthetic wording is supported only if ensemble conflict F1 improves by at
least 0.03 with the review-bootstrap interval above zero, at least two of three
seed-level pooled deltas are positive, macro-F1 and neutral F1 each decline by
no more than 0.01, and conflict false positives increase by no more than 0.01.
The interval conditions on fitted models; three seeds characterize but do not
eliminate training uncertainty. This polarity-only experiment cannot select a
new complete-system winner.

The run contains 30 fits. Each completed fit retains its best model and logits;
unfinished optimizer state remains resumable. The tmux wrapper streams its log,
writes an exit status, queues completion to the originating thread, and closes
automatically.
