# Research context and evidence boundaries

## Why publish V6

V6 contains the most developed continuous-update experiment interfaces: explicit evidence eligibility, bounded retrieval, structured proposals, typed arithmetic, own-state continuation, and cost-aware workflow comparisons. The public extraction packages those components for inspection. It does not declare V6 the best-performing model or release the full private experiment archive.

## Previous work, summarized

| Stage | Work actually performed | What it established |
|---|---|---|
| Tool recovery | Qwen2.5-0.5B LoRA SFT and episodic REINFORCE with a leave-one-out baseline (RLOO). | Learning improved the small model on bounded synthetic recovery, but rules solved the task and further SFT outperformed RL. |
| Financial evidence selection | Qwen2.5-1.5B SFT with frozen larger-model and metadata-rule controls. | Evidence selection improved on a small supplied-candidate task; this was not autonomous long-horizon research. |
| Financial note updates | 1.5B SFT, RLOO, revised terminal rewards and component-wise terminal rewards. | Real parameter updates occurred, but no stable generalization or cost advantage emerged from the short RL runs. |
| Reviewed 8B training | Three Qwen3-8B QLoRA arms: positive examples, corrective supervision, and original-data control. | Each completed 72 optimizer updates. The 48 distinct tasks expanded to 96/144/96 supervision records; these were not independent task counts. |
| V6 investigation and continuation | Reused an existing 8B SFT adapter and compared retrieval, reasoning, revision, and state-update workflows. | No new V6 training. Strong simple baselines materially changed the interpretation of apparent SFT benefits. |

The earlier 8B adapter primarily learned final-answer and update behavior. Its reviewed-positive targets did not teach complete search/open/check action trajectories. Calling that training long-horizon agent post-training would overstate the work. Historical training scripts, datasets, model weights, and detailed runs are not part of this public release.

## Selected historical observations

These are author-reported aggregate observations from the historical experiment archive. The source datasets and raw model outputs are excluded here; this public package alone cannot independently reproduce these empirical numbers.

- In the small continuous-update cohort (two issuers, eight claim endpoints, three successive disclosures), no-example RAG achieved 87.0–89.1% atomic evidence-requirement coverage and 5/8 completely supported endpoints. The reused SFT configuration achieved 80.7% coverage and the same 5/8 endpoint count. These are coverage measures, not investment accuracy.
- Incremental updating reduced logical calls from 24 to 18 relative to its matched examples/tools workflow with the same observed score. Exact-response sharing was present, and the cohort is too small for a general savings claim.
- On four frozen new-company investigation tasks, no evaluated configuration completely satisfied every required element. Native thinking took about 2.72 times the observed model-call time of the wide-context baseline without a consistent evidence-coverage gain.
- AI source reviewers were masked to method identity, but they were not independent human financial experts. Ranges across their scores are not statistical confidence intervals.

## What remains open

1. A model choosing the next useful source is different from a framework merely exposing a search action.
2. Better source/period/unit normalization and calibrated evaluation should precede training against a reward.
3. Genuine multi-step evidence-action SFT is a separate experiment from answer-only SFT.
4. Larger models, new domains, human review, and end-to-end analyst timing require fresh matched tests.
5. Commercial-product superiority requires a direct, fair comparison, not a comparison against a product description.

The public framework preserves these uncertainties instead of turning a negative or mixed result into a success claim.
