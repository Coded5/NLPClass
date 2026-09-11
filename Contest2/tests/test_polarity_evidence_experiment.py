from __future__ import annotations
import unittest
import torch
from zeroshot_classifier.data import LabeledRow
from zeroshot_classifier.polarity_evidence_experiment import (Calibration, adjusted_logits,
    bootstrap_stability, combined_loss, deterministic_eligible, evidence_class_scores,
    evidence_positive_weights, evidence_target)

class TestPolarityEvidence(unittest.TestCase):
    def test_targets(self) -> None:
        self.assertEqual([evidence_target(x) for x in ('positive','negative','neutral','conflict')],
                         [(1.,0.),(0.,1.),(0.,0.),(1.,1.)])

    def test_scores_decode_combinations(self) -> None:
        logits = torch.tensor(((8.,-8.),(-8.,8.),(-8.,-8.),(8.,8.)))
        self.assertEqual(evidence_class_scores(torch, logits).argmax(-1).tolist(), [0,1,2,3])

    def test_adjustment(self) -> None:
        output = adjusted_logits(torch, torch.zeros((1,4)), torch.tensor(((8.,8.),)), Calibration(1.,1.,0.,.5))
        self.assertEqual(output.argmax(-1).item(), 3)

    def test_combined_loss_backpropagates(self) -> None:
        polarity = torch.zeros((2,4), requires_grad=True); evidence = torch.zeros((2,2), requires_grad=True)
        total, classification, ev_loss = combined_loss(torch, polarity, evidence, torch.tensor((0,3)),
            torch.tensor(((1.,0.),(1.,1.))), torch.ones(4), torch.ones(2), .5)
        total.backward(); self.assertGreater(classification.item(), 0); self.assertGreater(ev_loss.item(), 0)
        self.assertIsNotNone(polarity.grad); self.assertIsNotNone(evidence.grad)

    def test_positive_weights(self) -> None:
        rows = [LabeledRow(str(i),'x','service',label,i) for i,label in enumerate(('positive','neutral','negative','conflict'))]
        self.assertEqual(evidence_positive_weights(torch, rows).tolist(), [1.,1.])

    def test_protect_both_rule(self) -> None:
        baseline = {'pair_micro_f1':.8,'gold_aspect_polarity_classes':{'neutral':{'f1':.4},'conflict':{'f1':.3}}}
        candidate = {'pair_micro_f1':.795,'gold_aspect_polarity_classes':{'neutral':{'f1':.4},'conflict':{'f1':.31}}}
        self.assertTrue(deterministic_eligible(candidate, baseline, .005))
        candidate['gold_aspect_polarity_classes']['conflict']['f1'] = .29
        self.assertFalse(deterministic_eligible(candidate, baseline, .005))

    def test_identical_bootstrap_predictions_pass(self) -> None:
        rows = [LabeledRow('1','x','service','neutral',0),LabeledRow('2','y','price','conflict',1)]
        predictions = [('1','service','neutral'),('2','price','conflict')]
        result = bootstrap_stability(rows,predictions,predictions,100,.005,seed=1)
        self.assertTrue(all(value['pass_rate'] == 1 for value in result.values()))

if __name__ == '__main__': unittest.main()
