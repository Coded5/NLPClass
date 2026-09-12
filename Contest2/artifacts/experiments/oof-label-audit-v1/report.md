# Out-of-fold ABSA label audit

This diagnostic uses saved historical out-of-fold predictions and only IDs from the original training split.
Aspect and polarity predictions came from different CV experiments, so their error rates are not directly comparable.

## Coverage

| Bucket | Full error pool | Reviewed sample |
|---|---:|---:|
| aspect_multi_misc | 57 | 20 |
| aspect_single_misc | 193 | 15 |
| aspect_other | 208 | 15 |
| polarity_conflict | 61 | 20 |
| polarity_neutral | 111 | 20 |
| polarity_other | 321 | 10 |

## First-pass assessments

| Assessment | Count |
|---|---:|
| clear model error | 25 |
| ambiguous annotation | 63 |
| suspected annotation error | 8 |
| insufficient evidence | 4 |

| Bucket | Clear model error | Ambiguous | Suspected annotation error | Insufficient |
|---|---:|---:|---:|---:|
| aspect_multi_misc | 6 | 14 | 0 | 0 |
| aspect_single_misc | 5 | 9 | 1 | 0 |
| aspect_other | 4 | 8 | 1 | 2 |
| polarity_conflict | 5 | 11 | 4 | 0 |
| polarity_neutral | 0 | 16 | 2 | 2 |
| polarity_other | 5 | 5 | 0 | 0 |

Assessment complete: **true**.

The selected sample is deliberately enriched for difficult minority cases. Its category rates do not estimate dataset-wide label quality.

## Findings and next step

The current assistant review assigns 25 of 100 cases to clear model error, 63 to ambiguous annotation, 8 to suspected annotation error, and 4 to insufficient evidence. These are provisional judgments, not adjudicated labels.

## Revision standard

The original 37/30/32/1 breakdown was too confident. Revision 2 preserves previous categories in the worksheet. No annotation manual was located in the repository search, so these criteria are a provisional review rubric, not official dataset rules.

- Mentioning food or money alone does not establish a separately evaluated food or price aspect.
- Tie sentiment to its target; opposite sentiments about different aspects do not automatically constitute conflict.
- Lack of an explicit overall judgment does not invalidate anecdotes/miscellaneous: personal narrative may belong there.
- Questions, comparisons, and implicit evaluations can convey sentiment. Missing context does not automatically establish neutral.
- Use ambiguous annotation when plausible readings remain; this category does not mean that the gold label is wrong.
- Reserve suspected annotation error for a strong textual mismatch and keep any alternative as a hypothesis requiring adjudication.

Clear examples still show missed secondary aspects (IDs 3 and 181), contrasting evaluations of the same aspect (IDs 763, 1842, and 1903), and sentiment transferred from ordering difficulty to food (ID 484). These observations support a clause-to-aspect attribution weakness, but do not establish its prevalence or prove the cause of the performance plateau.

**Next step:** obtain the original annotation guidance and independently review the uncertain cases, ideally blinded to predictions. Keep proposed alternatives out of training until adjudicated. This selected sample cannot establish dataset-wide label noise or justify concluding that annotation quality caused the plateau.

## Provenance limitations

- Aspect OOF models used only the original training split.
- Polarity OOF models used the original training and validation splits; this audit filters their held-out predictions back to training IDs.
- Saved class probabilities were unavailable. Error persistence counts configurations that repeat an error and must not be described as confidence.
- The historical test split is not read by this audit.
