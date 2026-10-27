"""Verify that only CUDA minibatches use pinned, asynchronous host transfers."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import offline_models as models


class LSTMTransferTests(unittest.TestCase):
    def _exercise_transfer(self, device):
        positions = np.arange(72, dtype=float)
        frame = pd.DataFrame({
            "time": pd.date_range("2025-01-01", periods=72, freq="h"),
            "actual_kwh": 20 + np.sin(positions / 3),
            "e_price": 0.3 + 0.1 * (positions % 24 > 12),
        })
        params = models._parameters("lstm", {
            "input_hours": 3, "output_hours": 2, "lstm_device": device,
            "lstm_hidden_size": 4, "lstm_batch_size": 16, "lstm_head_dropout": 0.0,
        })
        state = models._fit_preprocess("lstm", frame.iloc[:48], params)
        loader_calls, transfers = [], []
        original_loader = torch.utils.data.DataLoader
        original_to = torch.Tensor.to

        def data_loader(*args, **kwargs):
            loader_calls.append(kwargs.copy())
            return original_loader(*args, **kwargs)

        def transfer(tensor, *args, **kwargs):
            if "non_blocking" in kwargs:
                transfers.append((str(tensor.device), tensor.is_pinned(), args[0], kwargs["non_blocking"]))
            return original_to(tensor, *args, **kwargs)

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with patch.object(torch.utils.data, "DataLoader", side_effect=data_loader), \
                 patch.object(torch.Tensor, "to", transfer):
                model, best, history = models._fit_lstm(
                    frame, np.arange(3, 47), np.arange(48, 71), state, 1, "train")
        finally:
            torch.set_num_threads(previous_threads)
        use_cuda = device.startswith("cuda")
        self.assertEqual(best, 1)
        self.assertEqual(len(history), 1)
        self.assertEqual(next(model.parameters()).device.type, "cuda" if use_cuda else "cpu")
        self.assertEqual(len(loader_calls), 2)
        self.assertTrue(all(call["pin_memory"] is use_cuda for call in loader_calls))
        # Three training and two validation batches, history/future/label tensors each.
        self.assertEqual(len(transfers), 15)
        for source, pinned, requested, non_blocking in transfers:
            self.assertEqual(source, "cpu")
            self.assertEqual(requested, device)
            self.assertIs(pinned, use_cuda)
            self.assertIs(non_blocking, use_cuda)

    def test_cpu_batches_keep_synchronous_unpinned_path(self):
        self._exercise_transfer("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_training_and_validation_use_pinned_nonblocking_batches(self):
        self._exercise_transfer("cuda:0")


if __name__ == "__main__":
    unittest.main()
