# Public extraction record

The public framework is an extraction of the V6 continuous-update implementation, not a byte-for-byte publication of the historical experiment directory.

## Preserved

- Five core algorithm modules: evidence tools, typed evidence programs, continuous workflows, active acquisition, and full-source refresh.
- The experimental flag-and-carry note-state policy, source eligibility rules, explicit model-job boundary, and mechanical-versus-semantic distinction.
- Synthetic tests that exercise current code behavior without research datasets, models, or network access.

## Changed for publication

- Converted internal imports into a normal `bluebridge` Python package.
- Removed implicit loading of frozen few-shot research data; callers can explicitly supply optional examples.
- Replaced links to undistributed historical source archives with public implementation-version receipts.
- Generalized the research-note prompt from a particular industry to company research notes.
- Added documentation, three editable diagrams, a fictional in-memory demonstration, usage restrictions, and a release content checker.

Four tests that specifically checked undistributed historical archives are excluded and listed in `tests/README.md`. They are not reported as passed. Current synthetic behavior and the new public API are tested separately.

## Deliberately not included

No datasets, silver labels, reviewer packets, source-document copies, real-company model outputs, private correspondence, interviews, weights, credentials, machine launchers, training scripts, or historical run archives are distributed. The current package cannot independently reproduce the historical empirical tables without separately authorized inputs and compute.

The demonstration's invented figures are constructed within Python code. They are interface fixtures, not a research corpus or a new evaluation result.
