import unittest

import torch

from loss import grpo_clipped_loss


class GrpoLossTest(unittest.TestCase):
    def test_zero_advantage_has_zero_policy_loss(self):
        values = torch.zeros((2, 3))
        loss = grpo_clipped_loss(
            values,
            values,
            torch.zeros(2),
            torch.ones_like(values),
            max_completion_length=3,
        )
        self.assertEqual(loss.item(), 0.0)

    def test_mask_and_dr_grpo_normalizer(self):
        current = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        mask = torch.tensor([[1, 0], [1, 1]])
        loss = grpo_clipped_loss(
            current,
            current,
            torch.tensor([1.0, 1.0]),
            mask,
            max_completion_length=2,
        )
        self.assertAlmostEqual(loss.item(), -0.75)

    def test_clips_positive_advantage_ratio(self):
        current = torch.tensor([[1.0]])
        old = torch.tensor([[0.0]])
        loss = grpo_clipped_loss(
            current,
            old,
            torch.tensor([1.0]),
            torch.ones_like(current),
            max_completion_length=1,
            epsilon=0.2,
        )
        self.assertAlmostEqual(loss.item(), -1.2, places=6)


if __name__ == "__main__":
    unittest.main()
