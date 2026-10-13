# T2C-AD — Ambiguity-aware Text-to-Cypher

A multi-agent Text-to-Cypher pipeline that **detects and resolves ambiguity explicitly**
before generating a Cypher query, plus a typed ambiguity benchmark for evaluating it.

Built for an Honours thesis at the University of Western Australia (2026).

> **Research question:** To what extent does an ambiguity-aware Text-to-Cypher system improve
> semantic correctness compared to single-pass translation, on investigative-style questions?

Most Text-to-Cypher systems commit to a single reading of a question during schema linking
and never record that a choice was made. T2C-AD makes that choice a named pipeline stage.
The schema linker returns a *distribution* over candidate schema patterns, an Ambiguity
Detector decides whether the question is ambiguous (schema / entity / intent / temporal),
and a Disambiguator commits to one interpretation before query generation.

## Pipeline

```
NL question
   │
   ▼
Schema Linker ─────────▶ candidate distribution (diverse beam search, not argmax)
   ├─▶ Entity Lookup ──▶ matching KG instances (fuzzy match against live Neo4j)
   ▼
Bayesian Scorer ───────▶ schema / entity entropy
   ▼
Ambiguity Detector ────▶ ambiguous? which type?   (LLM judgement + entropy as evidence)
   ▼
Disambiguator ─────────▶ one committed SchemaMapping
   ▼
Query Generator ───────▶ Cypher
   ▼
CyVer Validation ──────▶ syntax fail → Query Generator (≤2)
   │                     schema/property fail → Schema Linker (≤3)
   ▼
DB Executor → Semantic Evaluator ── wrong result → Disambiguator (≤1)
```

Every LLM stage sits behind one `BaseLLM` interface, so you switch between local
HuggingFace + LoRA models (Mistral-7B, Qwen2.5-32B/72B, Llama-3.1-8B) and API models
(Claude, GPT-4o) in config. You don't need to change any code.

### Evaluation conditions

| Condition | What runs |
|---|---|
| **C1** | Zero-shot baseline: single-pass generation from the schema |
| **C2** | Schema-grounded: SFT schema linker + query generator, commits silently to the top candidate |
| **C3** | C2 + Ambiguity Detector + Disambiguator + the disambiguation feedback loop |

Metrics: execution accuracy (EX, primary), ambiguity-aware EA, Detection F1,
disambiguation success rate (DSR), validity rates, and paired bootstrap CIs.

## Benchmarks and graphs

| File | Graph | Items |
|---|---|---|
| `data/benchmark-v3.json` | `data/SyntheticPoliceKG-v3.cypher` (6 labels, 11 rel types) | 120 (80 ambiguous, 20 per type) |
| `data/benchmark-pole-external.json` | [`neo4j-graph-examples/pole`](https://github.com/neo4j-graph-examples/pole) | 100 (80 ambiguous, 20 per type) |
| `data/benchmark-updated.json` | `data/SyntheticPoliceKG.cypher` (v2, frozen baseline) | 125 (50 ambiguous) |

Each item has a question, an ambiguity type (or `null`), a default interpretation, and
for ambiguous items a gold Cypher query for every valid interpretation.

## Headline results

Execution accuracy (%), C1 / C2 / C3:

| Model | v3 synthetic | External POLE |
|---|---|---|
| Claude Sonnet 4.6 (API) | 7.5 / 37.5 / **52.5** | 20.0 / 59.0 / **64.0** |
| Qwen2.5-32B (LoRA) | 5.8 / 20.0 / **25.0** | 40.0 / 37.0 / **53.0** |
| Qwen2.5-72B (LoRA) | 5.8 / 19.2 / **20.8** | 41.0 / 36.0 / **43.0** |
| Mistral-7B (LoRA) | 3.3 / 10.8 / 9.2 | 14.0 / 21.0 / **24.0** |

Full tables are in `results/tables_*.md`. Per-question records are in
`results/qual_records_*.json`, and bootstrap CIs are in `results/bootstrap_ci.json`. These
are single runs, so check the CIs before reading a small difference as a real effect.

## Setup

Requires Python ≥ 3.11 and Neo4j 5.x (Docker or Neo4j Desktop).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

**Load a graph.** Create a database and run the matching Cypher file. For example, load
`data/SyntheticPoliceKG-v3.cypher` for the v3 benchmark. For the external benchmark, load
the [POLE example graph](https://github.com/neo4j-graph-examples/pole).

**Set credentials:**

```bash
export NEO4J_PASSWORD=...
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY, for API-backed runs
```

The Neo4j URI, user and database name are set in `config/pipeline*.yaml`.

## Running

```bash
# Full C1/C2/C3 evaluation for one config (writes results/metrics_*, tables_*, qual_records_*)
python -m experiments.run_evaluation config/pipeline.yaml
python -m experiments.run_evaluation config/pipeline_pole_external.yaml

# Qualitative C2-vs-C3 report from a run
python -m experiments.analyze_qualitative results/qual_records_<model>_<tag>.json

# Paired bootstrap confidence intervals over all runs
python -m experiments.bootstrap_ci

# Validate a benchmark against a live graph (syntax, schema closure, non-empty results, ...)
python -m experiments.validate_benchmark

# Tests (offline: no Neo4j or GPU needed)
pytest
```

API models need no GPU: set `active_model: claude-sonnet` in the config, and the same model
runs every stage. Local models also need the LoRA adapters for the schema linker and query
generator, saved at `checkpoints/<model_key>/{sl,qg}_adapter`. Train them with
`experiments/train_schema_linker.py` and `experiments/train_query_generator.py` (the configs
are in `config/`). Adapters are not included in this repo.

## Repository layout

```
pipeline/       the pipeline package: one subpackage per stage, shared types in types.py
experiments/    evaluation harness, condition builders, SFT training, probes, analysis
config/         model registry, pipeline run configs, training configs
data/           knowledge graphs (.cypher) and benchmarks (.json)
results/        metrics, tables and per-question records behind the thesis
kaya/           SLURM job scripts for UWA's Kaya HPC cluster (see kaya/README.md)
tests/          pytest suite (offline)
context/        specs and governance docs used to build the project (see below)
```

### About `context/`

The pipeline was built one step at a time with an AI coding agent (Claude Code), from
written specs. `context/` holds that process: the project overview, a spec per step
(`feature-specs/`), the progress tracker, and a decisions log that records every
deviation from a spec and why. `CLAUDE.md` is the agent's entry point.

## Acknowledgements

Supervised by Assoc. Prof. Tim French and Assoc. Prof. Wei Liu (UWA). Query validation
uses [CyVer](https://github.com/gmandilara/CyVer). Schema linker and query generator
training uses the [Neo4j Text2Cypher 2025v1](https://huggingface.co/datasets/neo4j/text2cypher-2025v1)
dataset.

## Licence

MIT. See [LICENSE](LICENSE).
