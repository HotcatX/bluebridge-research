"""Fictional, in-memory interface demo. No dataset, network, model or training."""
from __future__ import annotations

import json

from bluebridge import initial_state, prepare_event, reduce_event


def fictional_episode():
    """Build a tiny invented fixture; these figures describe no real company."""
    return {
        "episode_id": "fictional-interface-demo",
        "issuer": "Example Company (fictional)",
        "initial_as_of": "2025-01-01",
        "sources": [
            {"source_id": "old", "published_at": "2025-01-01"},
            {"source_id": "first", "published_at": "2025-04-01"},
            {"source_id": "second", "published_at": "2025-07-01"},
        ],
        "chunks": [
            {"chunk_id": "old-fact", "source_id": "old", "text": "Historical revenue was USD 100 million."},
            {"chunk_id": "first-fact", "source_id": "first", "text": "First quarter revenue was USD 110 million."},
            {"chunk_id": "second-fact", "source_id": "second", "text": "Second quarter revenue was USD 120 million."},
        ],
        "initial_claims": [
            {"claim_id": "historical", "instruction": "Keep this fixed historical revenue fact.",
             "text": "Historical revenue was USD 100 million.", "citations": ["old-fact"],
             "metric_spec": {"temporal_policy": "fixed_historical"}},
            {"claim_id": "latest", "instruction": "Update the latest quarter revenue.",
             "text": "Latest reported revenue was USD 100 million.", "citations": ["old-fact"],
             "metric_spec": {}},
        ],
        "events": [
            {"event_id": "first-arrival", "as_of": "2025-04-01", "new_source_ids": ["first"]},
            {"event_id": "second-arrival", "as_of": "2025-07-01", "new_source_ids": ["second"]},
        ],
    }


def scripted_outputs(prepared, episode, event_index):
    """Mock only: deliberately scripted answers, never an accuracy benchmark."""
    outputs = []
    for job in prepared["jobs"]:
        request = json.loads(job["messages"][-1]["content"])
        historical = request["claim_id"] == "historical"
        chunk = episode["chunks"][0 if historical else event_index + 1]
        proposal = {
            "tool": "finish", "disposition": "retain" if historical else "update",
            "value": 100 if historical else 110 + 10 * event_index,
            "unit": "USD million", "citations": [chunk["chunk_id"]],
            "reason": "Scripted fictional response for interface verification only.",
            "patch": chunk["text"],
            "evidence_quotes": [{"chunk_id": chunk["chunk_id"], "quote": chunk["text"]}],
            "calculations": [],
        }
        outputs.append({"job_id": job["job_id"], "text": json.dumps(proposal)})
    return outputs


def run_demo():
    episode = fictional_episode()
    summaries = []
    for arm in ("fixed_zero", "incremental_tools"):
        state = initial_state(episode)
        jobs = skipped = 0
        for event_index in range(len(episode["events"])):
            prepared = prepare_event(episode, state, event_index, arm)
            jobs += len(prepared["jobs"])
            skipped += len(prepared["skipped"])
            reduced = reduce_event(episode, state, event_index, prepared,
                                   scripted_outputs(prepared, episode, event_index))
            state = reduced["state"]
        if state["latest"]["text"] != "Second quarter revenue was USD 120 million.":
            raise RuntimeError("The fictional update did not propagate.")
        if any(claim["unresolved"] for claim in state.values()):
            raise RuntimeError("Unexpected mechanical flag in the scripted happy path.")
        summaries.append({"arm": arm, "prepared_jobs": jobs, "cached_claims": skipped,
                          "final_note": {key: value["text"] for key, value in state.items()}})

    # Separately demonstrate the original research flag-and-carry behavior.
    state = initial_state(episode)
    prepared = prepare_event(episode, state, 0, "incremental_selective")
    bad = scripted_outputs(prepared, episode, 0)
    proposal = json.loads(bad[0]["text"])
    proposal.update(value=999, patch="Latest revenue was USD 999 million.")
    bad[0]["text"] = json.dumps(proposal)
    flagged = reduce_event(episode, state, 0, prepared, bad)
    if not flagged["state"]["latest"]["unresolved"]:
        raise RuntimeError("An unsupported fictional value was not flagged.")
    return {
        "mode": "SCRIPTED OFFLINE DEMO — no model inference, no empirical performance claim",
        "workflows": summaries,
        "unsupported_value_demo": {
            "unresolved": flagged["state"]["latest"]["unresolved"],
            "flags": flagged["records"][0]["audit"]["flags"],
            "policy": "Flagged proposal remains visible; this is not an approval gate.",
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, ensure_ascii=False))
