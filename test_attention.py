#!/usr/bin/env python3

import torch
import attention


def test_attention():
    result = attention.run_single_attention() - torch.tensor(
        [[0.7311, 0.2689], [0.2689, 0.7311]]
    )
    assert torch.norm(result) < 0.0001
