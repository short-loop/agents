# fork/

Documentation for the ShortLoop patches carried on top of upstream
[livekit/agents](https://github.com/livekit/agents). Start with
[`../FORK.md`](../FORK.md): it has the fork facts, the patch index, the conflict
hot-spot map and the upstream sync playbook.

This folder contains **documentation only**: no code, no scripts. Nothing here is
imported or packaged.

## Layout

- `patches/NN-short-name.md` has one document per logical patch. The numbers are
  stable IDs. They are not an apply order; see "Patch dependencies" in FORK.md for that.

## Patch document layout

Every patch document uses the same sections, in this order, so an automated sync agent
can find what it needs:

1. **Header table**: status (Always-on / Opt-in), origin commits and fork PRs,
   dependencies, patches it shares code with, automated tests.
2. **Why**: the production problem and the evidence behind it.
3. **Behaviour**: what changes for users of the fork, defaults, configuration knobs,
   log lines (with exact message text and field names).
4. **Implementation walkthrough**: every file, class, function and attribute touched,
   and where in the upstream code the change sits.
5. **Re-applying the patch**: the intent, written so the patch can be re-implemented
   against a reshaped upstream file.
6. **Upstream contracts relied upon**: the upstream symbols and behaviours the patch
   depends on. A clean merge can still break these, so check them after every sync.
7. **Conflict guidance**: likely conflicts and how to resolve them.
8. **Verification after sync**: tests to run and things to inspect by hand.
9. **Known caveats**: quirks that are known and intended. Do not "fix" them during a
   sync.
10. **Drop criteria**: when upstream would make the patch unnecessary. Dropping a patch
    is always a separate, deliberate change.

## Adding or changing a patch

- New patch: take the next free number, write the document with the sections above,
  and add a row to FORK.md's patch index (section 2) and hot-spot map (section 3).
- Changed patch: update its document in the same PR as the code change, including the
  origin row (new fork commit or PR).
- Removed patch: keep the file, set the status to **Dropped**, and record the reason and
  the upstream change that replaced it. This keeps the history easy to follow for the
  sync workflow.
