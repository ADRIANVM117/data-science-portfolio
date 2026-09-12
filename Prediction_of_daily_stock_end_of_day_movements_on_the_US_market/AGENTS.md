# AGENTS.md

## Project purpose

This repository folder is a quantitative research project based on the
CFM / Challenge Data end-of-day stock movement prediction challenge.

Read `README.md` before making research-related changes.

The objective of this project is not only to maximize competition score.
It is also used to practice rigorous quantitative research and experimental
design.

---

## Research workflow

Do not immediately implement a new research idea.

For any new experiment:

1. Understand the research question.
2. Inspect the relevant existing code and notebooks.
3. Identify the current baseline and validation protocol.
4. Propose an implementation plan before making substantial changes.
5. Implement only after the experiment has been sufficiently specified.
6. Run the relevant checks/tests.
7. Report results and unexpected behavior.

Do not silently make methodological decisions when the specification is
ambiguous. Surface the ambiguity instead.

---

## Research integrity

Avoid data leakage and look-ahead bias.

Any transformation that learns parameters from data must be fitted using
training data only within the corresponding validation split.

Do not use the competition test set for:
- model selection,
- feature selection,
- hyperparameter selection,
- preprocessing decisions,
- hypothesis evaluation.

Do not change the target definition, validation scheme, evaluation metric,
or information set without explicitly flagging the change.

Do not select approaches solely because they improve in-sample performance.

Negative experimental results are valid results and should not be hidden.

---

## Data

The raw competition data must be treated as immutable.

Do not overwrite or manually modify raw files.

Missing values are meaningful until demonstrated otherwise.
Do not silently impute, drop, or replace them.

Preserve `ID`, `day`, and `equity` semantics described in `README.md`.

---

## Code organization

Reusable research logic belongs in `src/`.

Notebooks are used for:
- exploration,
- diagnostics,
- experiments,
- visualization,
- interpretation of results.

Avoid duplicating substantial preprocessing or modeling logic inside notebooks
when equivalent functionality belongs in `src/`.

Prefer extending existing modules over creating parallel implementations of
the same logic.

---

## Validation

Because observations have temporal and cross-sectional structure, do not
introduce random train/validation splits unless an experiment explicitly
requires and justifies them.

Any validation procedure must respect the information that would have been
available at prediction time.

Report train and validation/test-of-experiment results separately.

---

## Before modifying the project

Before substantial implementation:

1. Read `README.md`.
2. Inspect the relevant files in `src/`.
3. Inspect the notebooks related to the experiment.
4. Explain what files you intend to modify or create.
5. Explain why each modification is necessary.

Do not refactor unrelated parts of the project.

---

## Completion criteria

A task is not complete merely because code was written.

Before declaring completion:

- run the relevant code/tests when possible;
- verify imports and paths;
- check output shapes and missing values where relevant;
- verify that no obvious leakage was introduced;
- summarize files changed;
- report assumptions and unresolved issues;
- distinguish observed results from interpretation.

## Research context

Before substantial research work:

1. Read `README.md` and this file.
2. Read `docs/METHODOLOGY.md` for the current methodological state.
3. Read the recent relevant entries in `docs/RESEARCH_LOG.md`.
4. Read the active experiment contract in `experiments/`.

Treat these repository documents as the persistent source of truth.
Do not rely on previous chat/session context when it conflicts with
the repository.