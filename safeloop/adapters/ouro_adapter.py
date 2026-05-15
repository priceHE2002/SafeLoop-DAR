from __future__ import annotations

from safeloop.adapters.hf_depth_adapter import HFCausalDepthAdapter


class OuroAdapter(HFCausalDepthAdapter):
    """Ouro adapter.

    Ouro exposes recurrent-step controls through remote code/config fields. The
    base HF adapter sets `total_ut_steps=depth` and disables adaptive exit when
    probing fixed-depth states.
    """

