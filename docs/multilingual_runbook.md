# Multilingual Runbook — Answer Language

Short reference for the "answered in the wrong language" class of issues in the
MobiSystems RAG agent, what causes it, how it is currently fixed, and how to
extend the fix to more languages.

## Problem

The assistant sometimes replied in the wrong language:

- User asks in **English** → agent answers in **Bulgarian** (most common).
- Occasionally the reverse (Bulgarian question → English answer).

Reproduced even on a **fresh conversation with no Bulgarian history**, so it is
a generation-time language bias, not a history/context-carryover issue.

## Root cause

- The generation prompts (`_ANSWER_PROMPT`, `_DIRECT_PROMPT`, `_ELABORATE_PROMPT`)
  are written in **Bulgarian**, and much of the retrieved KB context is also
  Bulgarian.
- With an all-Bulgarian prompt + largely Bulgarian context, the model defaults
  to Bulgarian output.
- A single English instruction line (`_lang_directive`) nudges but does **not
  reliably override** that bias.

## Current fix

Two layers, both in `app/agent/nodes.py`:

1. **Instruction nudge — `_lang_directive(text)`**
   Explicitly orders the model to reply in the user's language; names English
   for pure-ASCII questions and Bulgarian for Cyrillic. Helps, but not a
   guarantee.

2. **Deterministic backstop — `_enforce_language(answer, question)`**
   After generation, compares the **script** of the question vs the answer via
   `_script()`:
   - Same script → return unchanged (no extra LLM call).
   - Mismatch → one cheap translation pass (`_TRANSLATE_PROMPT`, thinking
     disabled) into the user's language.

   Applied in `generate`, `elaborate`, and `generate_direct`.

   Rationale: in this **bilingual (BG/EN)** KB, a script mismatch (Cyrillic vs
   Latin) reliably means an EN↔BG mix-up.

## Verification (against the `mobisystems` collection)

- Standalone English `how can i send mail` → English answer.
- English question after Bulgarian history → English answer.
- Bulgarian question → Bulgarian answer.

Repro command:

```bash
QDRANT_COLLECTION=mobisystems DISABLE_UPLOAD=true python -c "
from app.agent.graph import ask
print(ask('how can i send mail', thread_id='t1')['answer'][:300])
"
```

## Known limitations

- The backstop distinguishes **Cyrillic (Bulgarian) vs Latin (English)** only —
  the actual scope of this KB.
- It cannot catch a **same-script** mismatch. E.g. a **French** question
  answered in **English** (both Latin) would not trigger a translation; only
  `_lang_directive` guides those cases.

## Plan: extend to more languages (French/German/Spanish/…)

Do this only if real non-English Latin traffic appears.

1. **Add a language detector** (e.g. `lingua-language-detector`, more accurate
   than `langdetect` on short text). Add to `requirements.txt`.
2. **Replace the script check** in `_enforce_language` with detected language
   codes: translate whenever `detect(answer) != detect(question)`.
3. **Pass the detected language name** into `_TRANSLATE_PROMPT` and
   `_lang_directive` so both use the same resolved target.
4. **Guard short/ambiguous text** — skip enforcement when detection confidence
   is low (avoid mistranslating one-word replies / product names).
5. **Verify** with a small multilingual test set (EN, BG, FR, DE, ES) in
   `app/eval/`.

## Related PRs

- #10 — initial `_lang_directive` nudge + provenance logging (insufficient
  alone).
- #14 — deterministic `_enforce_language` backstop (current fix).
