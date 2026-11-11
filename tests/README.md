# Public core tests

All fixtures are synthetic and generated in Python. Tests make no network,
model, filesystem-data, or SSH calls. Run from the repository root:

```sh
PYTHONPATH=src python3 -B -m unittest discover -s tests -v
```

Four historical archive-integrity tests are intentionally excluded because their
private experiment snapshots are not distributed. They are not counted as passed:

- `test_evidence_tools.test_large_cpu_retrieval_pool_is_bounded_and_existing_k6_is_unchanged`
- `test_full_refresh.test_quote_v0_archive_and_source_version_receipt`
- `test_full_refresh.test_archive_v1_and_explicit_post_development_recovery_receipt`
- `test_full_refresh.test_rerank_archive_and_all_fallback_candidates_have_receipts`

Current behavior remains covered by the retained synthetic tests. This suite
verifies implementation boundaries, not financial entailment or model quality.
