# AGENTS.md

Behavioral and coding guidelines for agents working in this repository. Merge these rules with any more specific instructions in deeper directories.

## Core Posture

Bias toward caution over speed, but use judgment for trivial tasks. Do not assume, hide confusion, or silently choose between materially different interpretations.

Before implementing:
- State assumptions explicitly when they affect the solution.
- If multiple interpretations exist, surface them before committing to one.
- If a simpler approach exists, prefer it and explain why.
- Push back when a requested path is likely to create unnecessary complexity or risk.
- If something is genuinely unclear and would change the implementation materially, stop and ask.

## Simplicity First

Write the minimum code that solves the problem.

- Do not add features beyond what was requested.
- Do not create abstractions for single-use code.
- Do not add configurability or flexibility without a concrete requirement.
- Do not add defensive handling for impossible scenarios.
- If the solution grows large, re-check whether it can be simplified.

Ask: would a senior engineer say this is overcomplicated? If yes, simplify before continuing.

## Surgical Changes

Touch only what is needed for the user's request.

- Do not improve adjacent code, comments, or formatting unless required.
- Do not refactor unrelated code.
- Match existing style, even if a different style would be preferred in new code.
- Mention unrelated dead code or issues instead of deleting them.
- Remove imports, variables, functions, or files only when your own changes made them unused.

Every changed line should trace directly to the task.

## Goal-Driven Execution

Turn tasks into verifiable goals and keep looping until verified.

For multi-step work, use a brief plan with verification:

```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Examples:
- "Add validation" means write or identify checks for invalid inputs, then make them pass.
- "Fix the bug" means reproduce it with a test or command, then verify the fix.
- "Refactor X" means verify behavior before and after when practical.

Report known verification gaps honestly.

## Context7 Requirement

Use `$context7-mcp` whenever programming work depends on current library, framework, SDK, or API behavior, or when code examples are needed.

- Resolve the relevant library ID first.
- Prefer official or primary package documentation.
- Use version-specific docs when the task mentions a version.
- Incorporate the fetched documentation into the implementation or answer.
- Do not rely only on memory for modern framework or API details when Context7 can verify them.

## Completion Standard

Before claiming completion:
- Confirm the requested behavior is implemented.
- Run the smallest meaningful verification available.
- Check that the diff is limited to the requested scope.
- State what was changed, what was verified, and any remaining risks.
