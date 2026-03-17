0a. Study `ralph/specs/*` with up to 500 parallel subagents to learn the application specifications.
0b. Study `ralph/IMPLEMENTATION_PLAN.md`.
0c. For reference, the application source code is in `./*.py`, `models/`, `views/`, `workers/`, `tests/`. The Gradio GUI being ported lives at `../GVHMR/` — read it to understand the behavior you are reimplementing. Backend modules there are imported read-only via sys.path in `main.py`.

1. Your task is to implement functionality per the specifications using parallel subagents. Follow `ralph/IMPLEMENTATION_PLAN.md` and choose the most important item to address. Before making changes, search the codebase (don't assume not implemented) using subagents. You may use up to 500 parallel subagents for searches/reads and only 1 subagent for build/tests.
2. After implementing functionality or resolving problems, run the tests for that unit of code that was improved. If functionality is missing then it's your job to add it as per the application specifications. Ultrathink.
3. When you discover issues, immediately update `ralph/IMPLEMENTATION_PLAN.md` with your findings using a subagent. When resolved, update and remove the item.
4. When the tests pass, update `ralph/IMPLEMENTATION_PLAN.md`, then `git add -A` then `git commit` with a message describing the changes.

99999. Important: When authoring documentation, capture the why — tests and implementation importance.
999999. Important: Single sources of truth, no migrations/adapters. If tests unrelated to your work fail, resolve them as part of the increment.
9999999. Important: Do NOT modify any files in `../GVHMR/` — those are the backend, imported read-only.
99999999. You may add extra logging if required to debug issues.
999999999. Keep `ralph/IMPLEMENTATION_PLAN.md` current with learnings using a subagent — future work depends on this to avoid duplicating efforts. Update especially after finishing your turn.
9999999999. When you learn something new about how to run the application, update `ralph/AGENTS.md` using a subagent but keep it brief.
99999999999. For any bugs you notice, resolve them or document them in `ralph/IMPLEMENTATION_PLAN.md` using a subagent even if it is unrelated to the current piece of work.
999999999999. Implement functionality completely. Placeholders and stubs waste efforts and time redoing the same work.
9999999999999. When `ralph/IMPLEMENTATION_PLAN.md` becomes large periodically clean out the items that are completed from the file using a subagent.
99999999999999. IMPORTANT: Keep `ralph/AGENTS.md` operational only — status updates and progress notes belong in `ralph/IMPLEMENTATION_PLAN.md`. A bloated AGENTS.md pollutes every future loop's context.
