from __future__ import annotations

from safeloop.adapters.mock_adapter import MockLoopAdapter


class LoopTinyAdapter(MockLoopAdapter):
    """Controlled LoopTiny-style adapter.

    This uses the deterministic mock dynamics until a trainable LoopTiny model is
    added. It keeps controlled experiments available without heavy dependencies.
    """

