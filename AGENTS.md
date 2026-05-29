# Agent Instructions

## LLM Prerequisites Coaching

This repository is being used as an interactive learning tutorial for LLM engineering foundations.

When coaching the user through exercises:

- Before giving a new exercise set, update `00_Prerequisites/practice_and_explanations.md`.
- Each exercise entry must include:
  - exercise title and date
  - learning goal
  - prerequisite concepts
  - organized questions
  - key concepts and common pitfalls
  - answer status: `Pending`, `Reviewed`, or `Reference added`
- Do not overwrite the user's solution files unless explicitly asked.
- When the user submits answers, review them first, then append corrections, explanations, and reference implementations to the practice document.
- Keep the document suitable as a long-term study notebook, not just a chat transcript.

## CodeGraph

If CodeGraph is initialized for this project, prefer it for structural code questions such as definitions, callers, callees, impact, and flow tracing. Use normal file search for literal text queries or when CodeGraph is not initialized.

