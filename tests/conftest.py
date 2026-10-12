from __future__ import annotations

import pytest

# A tiny dataset with the real one's quirks: a passage shared by two questions, a question
# asked twice of different passages (two golden), and a question repeated verbatim.
ROWS = [
    ("What did revenue do in 2023?", "Revenue rose 12% to $4.1 billion in 2023."),
    ("How much did revenue grow?", "Revenue rose 12% to $4.1 billion in 2023."),
    ("What are the risks?", "Supply chain disruption is a key risk."),
    ("What are the risks?", "Interest rate changes are a key risk."),
    ("Who is the CEO?", "Jane Doe has been CEO since 2019."),
    ("Who is the CEO?", "Jane Doe has been CEO since 2019."),
] + [(f"Question number {i}?", f"Passage number {i} about topic {i}.") for i in range(40)]


@pytest.fixture
def rows() -> list[tuple[str, str]]:
    return list(ROWS)
