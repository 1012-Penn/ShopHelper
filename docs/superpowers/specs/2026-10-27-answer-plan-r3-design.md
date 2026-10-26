# Evidence-Bound Answer Plan R3

## Goal

Improve citation-supported customer-service answers without treating an
existing citation as proof of semantic support. The optional path produces a
structured, evidence-bound answer plan, validates it against the exact candidate
evidence, and renders the verified plan. The original answer path remains the
default.

## Contract

- Inputs are the visible user question and actual numbered candidate evidence.
  Expected behavior, gold facts, scoring rules and reference answers are never
  sent to the planner.
- Every material claim binds `evidence_id`, an exact quote and its character
  span. Evidence source identity, entities and policy scope remain available.
- Deterministic validation rejects missing evidence IDs, quotes outside their
  evidence, entity/order conflicts, number/unit conflicts, policy-version or
  date conflicts, and omitted applicability conditions.
- Validation does not promote lexical overlap into semantic truth. A claim
  whose key assertion cannot be anchored remains `uncertain`; `uncertain` is
  never a strict success.
- A failed or uncertain plan may be repaired once. The repaired plan is
  revalidated. Insufficient plans render clarification, limited help or refusal;
  unsupported claims are not silently included.
- The renderer maps only verified claims to `[n]` citations, preserves
  conditions and limitations, and does not paste large evidence passages.

## Evaluation

The experiment uses a new frozen 8 dev + 20 held-out set split by source
template. Both arms receive the same model, question, evidence snapshot and
answer budget. The baseline produces free text; the treatment produces and
renders an answer plan. Scoring is isolated in a frozen deterministic rubric.
Provider usage is reserved before each call and settled afterward in a dedicated
SQLite ledger with a `$2` protection cap. Held-out is run once after freezing.

## Integration

The unified acceptance session adds an opt-in settings flag and graph node. This
branch does not modify `app/main.py`, `app/config.py`, existing graph/agent
wiring, or shared evaluation checks. The rollback is to disable the optional
node and leave the existing answer and verifier path unchanged.
