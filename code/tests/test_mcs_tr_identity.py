from __future__ import annotations

import numpy as np

from bnsv.mcs_dm import model_confidence_set


def test_mcs_is_explicitly_identified_as_tr_range_statistic() -> None:
    rng = np.random.default_rng(31)
    common = rng.normal(scale=0.1, size=160)
    losses = np.column_stack(
        [
            common + rng.normal(scale=0.02, size=160),
            common + 0.08 + rng.normal(scale=0.02, size=160),
            common + 0.80 + rng.normal(scale=0.02, size=160),
        ]
    )
    result = model_confidence_set(
        losses,
        ["best", "middle", "worst"],
        alpha=0.10,
        bootstrap_replications=400,
        mean_block_length=10,
        seed=17,
    )
    assert result.statistic == "T_R"
    assert result.elimination_order[0] == "worst"
