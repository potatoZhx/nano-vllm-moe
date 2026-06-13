# Multiple Cache Ratios for Random-Cache Collection

## Goal

Allow `collect_random_cache_acceptance.py` to collect one dataset containing
samples generated with different cache ratios. Each sample selects one cache
ratio according to user-provided weights, while existing single-ratio commands
continue to work.

## Command-Line Interface

`--cache-ratio` accepts one or more floating-point values:

```bash
--cache-ratio 0.25 0.5 0.75
```

The new optional `--cache-ratio-weights` argument accepts one positive weight
for each cache ratio:

```bash
--cache-ratio 0.25 0.5 0.75 \
--cache-ratio-weights 1 2 1
```

Weights are relative and do not need to sum to one. When weights are omitted,
all configured cache ratios receive equal weight. The existing command
`--cache-ratio 0.5` remains valid and behaves as before.

Validation rejects:

- Cache ratios outside `(0, 1]` or non-finite values.
- Non-positive or non-finite weights.
- A weight list whose length differs from the cache-ratio list.

## Sampling and Collection

At startup, the script normalizes the command-line values into parallel cache
ratio and weight lists. A dedicated `random.Random(args.seed)` instance selects
one ratio for each attempted sample using weighted random sampling. Keeping
this generator separate prevents unrelated random operations, such as Wiki
window selection, from changing the cache-ratio sequence.

The selected ratio is passed explicitly to `collect_one_sample`. Before draft
decoding, `build_all_expert_caches` reconfigures every wrapped MoE layer with
that ratio and builds the corresponding cache from the current sample's
prefill activations. The initial model wrapper uses the first configured ratio;
this is only initial state because each collected sample reconfigures it before
random-cache decoding.

Weighted random sampling is preferred over exact preallocated quotas because
collection failures can skip samples and invalidate an exact planned ratio
count. A fixed seed makes the attempted sequence reproducible, while large
datasets converge to the requested distribution.

## Output

Each record stores the selected scalar value in
`metadata.cache_ratio`. Existing downstream consumers therefore continue to
see the same field and type. `cache_summary` also reflects the selected ratio
because the model is reconfigured before the summary is generated.

Single-ratio runs retain the current output-directory naming:

```text
<dataset>_random_cache_<policy>_ratio0.5_topc0.5
```

Multi-ratio runs use a descriptive mixed-ratio name containing the configured
ratios and weights, so they do not collide with a single-ratio run. For
example:

```text
<dataset>_random_cache_<policy>_ratios0.25-0.5-0.75_weights1-2-1_topc0.5
```

## Error Handling

Configuration validation happens before output directories or model loading.
Invalid combinations fail immediately with an actionable `ValueError`.
Per-sample model or data failures retain the collector's current behavior:
print the traceback, skip that sample, and continue.

## Tests

Focused unit tests will cover:

- A single cache ratio with omitted weights.
- Multiple ratios with omitted weights receiving equal weights.
- Multiple ratios with explicit weights.
- Rejection of invalid ratios, invalid weights, and mismatched list lengths.
- Deterministic weighted selection with a seeded random generator.
- Output-directory naming for single- and multi-ratio configurations.

The existing Wiki article sampling tests will remain unchanged and run as
regression coverage. The README and script usage example will document both
single-ratio and mixed-ratio commands.
