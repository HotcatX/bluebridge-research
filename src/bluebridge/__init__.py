"""BlueBridge: provider-independent evidence and research-note workflows.

No models, credentials, external data or network clients are bundled. Mechanical
checks do not establish semantic correctness. Flagged proposals remain visible
and unresolved; applications must define their own publication/approval policy.
"""
from .workflows import initial_state, prepare_event, reduce_event, selective_jobs

__version__ = "0.1.0"
__all__ = ["initial_state", "prepare_event", "reduce_event", "selective_jobs"]
