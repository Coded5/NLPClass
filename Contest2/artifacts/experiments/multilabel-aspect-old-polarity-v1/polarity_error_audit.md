# Best polarity ensemble error audit

## Scope

This audit evaluates the selected three-seed inverse-frequency weighted-CE
polarity ensemble on the historical test split. Predictions come from
`test/oracle-aspects/predictions.csv`, so every input uses a gold aspect. The
analysis therefore isolates polarity classification from aspect detection.

The split contains 316 gold `(review ID, aspect)` rows. The model classified
264 correctly and 52 incorrectly, for accuracy `0.8354`. The complete composed
system has lower polarity micro-F1 (`0.8111`) because predicted aspects add
aspect-detection errors.

## Confusion matrix

Rows are gold labels and columns are predictions.

| Gold | Positive | Negative | Neutral | Conflict | Recall |
|---|---:|---:|---:|---:|---:|
| Positive | 176 | 6 | 5 | 0 | 0.9412 |
| Negative | 9 | 58 | 5 | 0 | 0.8056 |
| Neutral | 8 | 7 | 24 | 1 | 0.6000 |
| Conflict | 4 | 6 | 1 | 6 | 0.3529 |

The model predicted `conflict` only seven times. Six were correct, giving high
conflict precision (`0.8571`) but low recall (`0.3529`). Conflict F1 `0.5000`
therefore does not mean the model recognizes most mixed opinions; it mostly
avoids predicting that class.

Error rates by aspect were service `20.0%` (10/50),
anecdotes/miscellaneous `18.8%` (18/96), food `15.2%` (16/105), ambience
`13.5%` (5/37), and price `10.7%` (3/28).

## Why conflict is missed

All 11 missed conflict rows were reviewed. The explanations below are
hypotheses about the language pattern; they do not replace the supplied gold
labels.

| ID / aspect | Predicted | Text pattern | Likely difficulty |
|---|---|---|---|
| 174 / service | Negative | Slow in the evening, not a problem at lunch | Polarity changes with time; the first negative clause dominates. |
| 2189 / service | Neutral | “not exactly five star, but ... not ... a big deal” | Two negations soften a negative assessment. |
| 3000 / miscellaneous | Positive | Quotes another review's negative experience before rejecting it | The two opinions belong to different speakers. |
| 1035 / price | Negative | “overpriced but worth it” | Canonical negative-plus-positive concession; “overpriced” dominates. |
| 811 / ambience | Negative | Empty restaurant followed by a mitigating explanation | The positive side is implied rather than stated. |
| 1393 / food | Positive | Likes noodle dishes compared with green curry | Mixed sentiment is distributed across food items. |
| 1018 / food | Negative | Sometimes bad food, sometimes good food | Mixed sentiment is distributed across visits and interleaved with service. |
| 3114 / service | Negative | Long wait followed by an accommodating hostess | The improvement is an action rather than an explicit positive adjective. |
| 59 / ambience | Positive | Pub atmosphere and being good to children | The conflicting ambience judgment is subtle and may depend on context. |
| 594 / ambience | Negative | Some may like it; the writer found it annoying | Opposing opinions belong to different people. |
| 2366 / food | Positive | One diner liked one item; “that's it” implies broader dislike | The negative side is conveyed by restriction and context. |

Six of the misses became `negative`, four became `positive`, and one became
`neutral`. The recurring failure is collapsing a sentence to its strongest
explicit sentiment instead of preserving both aspect-specific opinions.

## Why neutral is missed

The model missed 16 of 40 neutral rows: eight became positive, seven became
negative, and one became conflict.

Several are factual, imperative, or incomplete statements whose surface words
resemble sentiment:

- `672 / food`: cooking and eating instructions became positive.
- `520 / food`: “Did I mention the wine?” became positive.
- `123 / miscellaneous`: “Ciao Bella” became positive.
- `792 / service`: “Has the chef and owner changed?” became negative.
- `1964 / price`: a factual `$160` bill became negative.
- `3450 / service`: a statement that the restaurant offers no dessert became
  negative.

Other neutral errors expose aspect leakage:

- `1213 / miscellaneous` contains explicit negative opinions about service,
  food, and price. The requested miscellaneous aspect is neutral, but the
  sentence-level negativity dominates.
- `594 / food` says food is “decent at best” while stronger mixed sentiment is
  about ambience. The model predicted conflict for food.
- `1857 / miscellaneous` includes positive space and a negative surprise whose
  intended target is unclear without surrounding context.

Some neutral labels are linguistically close to weak sentiment, including
“minimalist and clean - nothing to distract or commend,” “very spicy but not
offensive,” and “hungry a few hours later.” These cases make the neutral
boundary difficult even when the aspect is supplied.

## Positive and negative errors

Positive and negative performance is much stronger, but 25 errors remain.
Common patterns are:

- **Wrong-aspect distraction:** negative service examples mention great food,
  while positive service examples mention spotty service or lack of
  reservations. The model follows the strongest sentence-level sentiment.
- **Sarcasm and rhetorical language:** “Anybody who likes this place must be
  from a different planet,” “If you want Americanized Chinese food ... this is
  your place,” and praise of “fancy expensive ingredients” were predicted
  positive despite negative intent.
- **Implicit evaluation:** leaving, easy reservations, short waits, portion
  size, personal loyalty, and recommendations require pragmatic inference.
- **Negation and contrast:** “nothing was left,” “couldn't make up for,” and
  “nothing I would have again” are easy to reverse or weaken incorrectly.

At least two rows look questionable for the requested aspect based on the
isolated text: `2067 / service` is gold positive although it says service can
be spotty, and `2053 / service` is gold negative although its wording frames
the wait as worthwhile. These should be recorded as review candidates, not
automatically relabeled.

## Implications

More conflict weighting or oversampling is unlikely to fix the main errors by
itself. The model already learned a conservative conflict boundary, and the
five-fold oversampling experiment did not improve conflict reliably.

The next model experiment should make the requested aspect and label meaning
more explicit. Score four candidate statements such as “The opinion about
price contains both positive and negative evidence” against the review. Compare
that formulation with the existing weighted-CE model on the same grouped folds.
Report conflict precision and recall separately: improving recall by predicting
conflict everywhere would repeat the false-positive behavior observed in the
cross-validation models.

For error analysis, retain full review context when available and inspect all
annotations sharing an ID. The isolated row text sometimes contains sentiment
for several aspects or speakers, which is precisely where the classifier fails.
