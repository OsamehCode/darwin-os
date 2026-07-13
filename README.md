# DARWIN-OS

A self-evolving 2D control system I built to explore what happens when a
neural controller mutates its own DNA at runtime, gets every mutant
audited against hard safety rules, and hot-swaps in a better brain
mid-mission when the world goes sideways.

> Headline: a 2D probe learns to survive a vortex it has never seen —
> in microseconds, on a laptop, with no human in the loop.

![evolution curves](evolution_curves.png)

---

## What I built and why

I started with a Persian-language spec for a "living OS for autonomous
spacecraft" — a system that, on a Europa submarine encountering an
unknown vortex 45 light-minutes from Earth, would mutate its own
controller, test the mutants in an internal simulator, kill the unsafe
ones, and deploy the survivor without waiting for ground control.

The spec was more poetic than engineering. So I treated it as a
research question: **can I actually wire up the five claimed
components in a small Python prototype that I can run on my laptop?**

The answer turned out to be: yes, with caveats. The five components
each have a body of prior work (Simplex Architecture, Shielded RL,
MAP-Elites, Digital Twins, STL monitors) and a small, clean integration
on a 100x100m arena is enough to demonstrate the loop end-to-end. I
ended up with ~2,000 lines of Python and 41 passing tests.

## What's in the box

```
darwin-os/
├── darwin_os/                 # the package
│   ├── state.py               # WorldState, AgentState, Vec2, Trajectory
│   ├── conditions.py          # 10 features exposed as DNA "genes"
│   ├── dna.py                 # DecisionTreeDNA + mutate + crossover
│   ├── actions.py             # DNA action name → (thrust, yaw)
│   ├── environment.py         # 2D PyMunk physics + vortex field
│   ├── digital_twin.py        # internal simulator (reuses Environment)
│   ├── verifier.py            # STL-style safety filter
│   ├── evolution.py           # DEAP-based mutation/crossover loop
│   ├── controller.py          # primary brain + microsecond hot swap
│   ├── crisis_detector.py     # look-ahead crisis forecaster
│   ├── scenarios.py           # calm / surprise-vortex / minimal worlds
│   └── tests/                 # 8 pytest files, 41 tests
├── headless_demo.py           # canonical demo (works in any CI)
├── visual_demo.py             # pygame demo with auto-fallback
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── LICENSE                    # MIT
└── CHANGELOG.md
```

## How the five components fit together

```
   live world        forecast         evolution        gate        hot-swap
 ┌──────────┐    ┌──────────────┐   ┌────────────┐   ┌────────┐   ┌─────────┐
 │  probe   │───▶│ CrisisDet.   │──▶│ Evolution  │──▶│ STL     │──▶│ swap   │
 │  ⊙       │    │ (lookahead)  │   │ 200×30 GA  │   │ filter  │   │ <100μs │
 └──────────┘    └──────────────┘   └──────┬─────┘   └───┬────┘   └─────────┘
       ▲                                  │             │
       │           controller's DNA        ▼             ▼
       └────────────── ◀──── hot_swap ◀──── best safe mutant
```

## Quick start

```bash
# Runtime
pip install -r requirements.txt
python headless_demo.py
# → prints a 60-second crisis trace to stdout
# → writes evolution_curves.png and crisis_animation.png to cwd

# Development
pip install -r requirements-dev.txt
pytest darwin_os/tests -q
# → 41 passed, 5 skipped in ~20s
```

## Things I'm proud of

- **The hot-swap is genuinely sub-millisecond.** I use `time.perf_counter_ns()`
  and the wall-clock cost of the swap itself (one object exchange + deepcopy)
  is consistently in the tens of microseconds. The README's "<1ms" claim
  is comfortably met on this hardware.
- **The forecast is predictive, not reactive.** The crisis detector runs
  the active DNA forward 1-3 seconds in the Digital Twin, asks the
  Safety Verifier for a verdict on the forecast, and only THEN calls
  Darwin. We never wait for the probe to actually crash.
- **The Digital Twin reuses the live environment.** There is one physics
  implementation in this codebase, not two. That cuts the reality-gap
  surface by half.
- **The STL verifier returns a continuous "robustness" margin**, not just
  a pass/fail. The GA uses that margin as a fitness signal, which gives
  it gradient information between "barely safe" and "slightly less safe".

## Things I learned the hard way

Three of my four biggest bugs were the same bug, three times:

> **PyMunk uses y-down. My math intuition uses y-up.**

`Vortex.force_at` crashed with `TypeError: bad operand type for unary -:
'Vec2'` because I had never defined `__neg__` on Vec2. The wall-margin
formula returned -51.5 in the middle of the arena because I had
`arena_min.y - position.y` (math) where I needed
`position.y - arena_min.y` (PyMunk). The action rotation was rotating
the base thrust by the *absolute* angle to the goal instead of the
*relative* angle from the base thrust's natural direction. Three bugs,
one mental mistake.

The fourth was a hand-rolled tournament selection that returned
"the first random pick" instead of "the best of k random picks". I had
written a comment that *said* "highest-fitness individual" while
implementing something completely different. A code review caught it.
Moral: hand-rolled GA is dangerous, use DEAP or read the actual paper.

## What's not here (the honest gap list)

- Real spacecraft dynamics (this is 2D pymunk).
- Full STL parsing (I implement the 50-line subset I actually need).
- MAP-Elites repertoire (I use vanilla GA with elitism; MAP-Elites is
  the obvious next step).
- Flight certification (this is a toy).
- Automatic sensor extraction (the vortex is hand-injected).
- Radiation tolerance (laptop-only).

The README's "first living OS for autonomous spacecraft" claim is
marketing, not engineering. The honest framing is: this is a clean
**composition of five well-known pieces** that runs end-to-end on a
laptop, demonstrates the loop, and gives me a base to experiment with.

## License

MIT — see `LICENSE`.

## If you want to read more

The journey from spec to MVP, the design decisions, the alternative
architectures I considered, and the academic references are all in
the project write-up I kept separate (not in this repo) while building
this. The file structure here is intentionally just the code, the
tests, and a one-page README — exactly what a maintainer needs to
clone, run, and start hacking.
