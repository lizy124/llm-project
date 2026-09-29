## Motivation

KV pooling (AscendStoreConnector, both `memcache` and `mooncake` backends, with/without `use_layerwise`) currently has **no nightly coverage**: a pooling regression would not be caught until a manual weekly/CI run happens. This PR brings KV pooling into the DeepSeek-V4-Flash nightly guardrails by **overlapping pooling config onto 3 existing nightly services** — no new yaml files, no new test cases, no CI/runner script changes. Accuracy + performance are both guarded.

The same integration shape is already proven by two merged weekly cases: `DSv4-Flash-Memcache-PD-layerwise` (multi-node PD, memcache) and `Qwen3.5-397B-Memcache-perf` (single-node, memcache). This PR reuses their structures verbatim.

## Changes

[Superseded by pr_body_current.txt -- 4-entry version with S3, as pushed to PR #17624]

Only the last commit `497ff0482` (+82/-3, 3 nightly config yamls). The branch is based on current upstream `main` (`abe8ff18a`).

| Nightly entry | Platform / topology | Pooling overlay | Guarded by |
|---|---|---|---|
| `DeepSeek-V4-flash-w8a8-PD-prefix` (multi-node PD) | a3, prefill DP4xTP4 + decode DP16xTP1 | yaml-level `kv_pool` (memcache, 5000/6000); both ends switch `MooncakeHybrid` -> `MultiConnector` (PD transfer kept as connectors[0] with `engine_id` preserved; `AscendStoreConnector` memcache producer `use_layerwise: true` / consumer `false`) | `acc_aime2025` (90) + appended `perf_TPOT50_3_5_1_5` (15569.48 / 0.95), aligned with the weekly case |
| `DeepSeek_V4_Flash_A5` case #1 only | a5, single node DP8 | per-case `kv_pool` (mooncake, master 50088 / metrics 50089) + `AscendStoreConnector` `kv_both` `use_layerwise: true`; benchmarks expanded inline (decoupled from the shared anchor) | `acc` GPQA (87.4) + `perf` gsm8k; the 04/05/06 rejection-sampler comparison matrix and all other cases are untouched |
| `acceptace_rate_deepseekv4-flash-w8a8-mtp` (MTP3) | a3, single node TP8 | `_kv_pool` memcache anchor + `AscendStoreConnector` `kv_both` `use_layerwise: false` appended to the shared `_server_cmd` (Qwen weekly precedent); both open/close cases carry the pool | acceptance rate (0.925 ± 0.05) + `perf` (90) |

Ports follow the nightly conventions (memcache 5000/6000, mooncake master 50088 / metrics 50089); memcache pool services are per-case lifecycle on single-node runners and node-0 managed on the multi-node runner.

`kv_load_failure_policy` is deliberately left at the default `fail` (loud failure) everywhere — silent recompute would mask exactly the read-path bugs this guardrail exists to catch.

## Local static validation

All 3 yamls pass `yaml.safe_load`; every JSON server arg passes `json.loads`; both `kv_pool` blocks round-trip through the real parser (`tests/e2e/common/kv_pool/config.py`).

## Test plan

- [ ] `/nightly DeepSeek-V4-flash-w8a8-PD-prefix` — expect AIME 90 unchanged; TPOT50 ~15569 (write-path overhead watermark)
- [ ] `/nightly DeepSeek_V4_Flash_A5` — expect GPQA 87.4 unchanged; gsm8k perf re-baselined after first pooled run; remaining 5 cases byte-identical behavior
- [ ] `/nightly acceptace_rate_deepseekv4-flash-w8a8-mtp` — expect acceptance rate 0.925 unchanged; perf re-baselined
- [ ] Pool service logs: memcache/mooncake start-stop clean per case; `kvpool hit tokens` expected to stay 0 (no shared prefixes in these workloads — query path exercised, hit path is not)

## Rollback

Fully reversible: removing the `kv_pool` block and the `--kv-transfer-config` entries restores the original nightly configs verbatim.
