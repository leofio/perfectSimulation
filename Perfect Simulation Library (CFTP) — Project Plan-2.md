# Perfect Simulation Library (CFTP) — Project Plan

2026-09-18 · @Leo

## 1. Why perfect simulation

Burn-in error is not just unknown, it is unknowable from the sample path. Every diagnostic — Gelman–Rubin, effective sample size, trace plots — is a function of states already visited, so none can detect probability mass in a region never reached.

The canonical failure, and Figure 1 of the docs: 2D Ising below the critical temperature with single-site Glauber started from all-plus. The chain looks converged; the stationary distribution is symmetric under global spin flip and you have sampled exactly half of it. R-hat will be fine.

Perfect simulation removes the question. The output is an exact draw from π, and successive draws are i.i.d., so Monte Carlo error bars are the elementary ones with no autocorrelation correction and ESS = N exactly.

## 2. The CFTP theorem

Write the chain as a stochastic recursive sequence, X(n+1) = f(X(n), U(n)) with U(n) i.i.d. This defines random maps φ(n) = f(·, U(n)) on the state space S, each preserving π.

Two compositions of the same maps behave completely differently:

- **Forward:** Ψ = φ(T−1) ∘ … ∘ φ(0). Converges in distribution, never pathwise — it keeps moving forever.
- **Backward:** Φ = φ(−1) ∘ φ(−2) ∘ … ∘ φ(−T). **Eventually constant almost surely.**

Once Φ collapses to a single point, extending further back changes nothing, because Φ(−T′ → 0) = Φ(−T → 0) ∘ Φ(−T′ → −T) and the outer map is constant. So set T\* = inf{T : Φ(−T → 0) is constant} and output its common value Y. Since Y is the limit of Φ(−T → 0)(x) for any x, and Φ(−T → 0) applied to a π-distributed state is π-distributed, Y \~ π **exactly**. No approximation, no error term.

### 2.1 Formal statement

**Setup.** Let S be a finite set, P an irreducible aperiodic transition matrix with stationary distribution π. A *random map representation* of P is a probability measure on maps S → S such that a random map φ drawn from it satisfies P(φ(x) = y) = P(x, y) for all x, y in S. Let (φ\_t) for t in ℤ be i.i.d. draws from it, and for s ≤ t write

> Φ(s → t) = φ(t−1) ∘ φ(t−2) ∘ … ∘ φ(s),   with Φ(t → t) = id.

Set T\* = inf{T ≥ 0 : Φ(−T → 0) is a constant map}.

**Theorem (Propp–Wilson, 1996).** Suppose P(Φ(−T → 0) is constant) → 1 as T → ∞. Then

1. T\* is almost surely finite, and P(T\* > mT) ≤ P(T\* > T)^m for every m, so T\* has exponentially decaying tails and all moments finite;
2. the value Y := Φ(−T\* → 0)(x), which does not depend on x, is well defined;
3. Y \~ π **exactly**.

**Proof.** *Nesting.* For T′ ≥ T, Φ(−T′ → 0) = Φ(−T → 0) ∘ Φ(−T′ → −T). So if Φ(−T → 0) is constant with value y, then Φ(−T′ → 0) ≡ y for every T′ ≥ T. Hence Φ(−T → 0)(x) = Y for all T ≥ T\* and all x, and (2) holds. The tail bound in (1) follows because the events that the m disjoint blocks of length T each fail to coalesce are independent, and failure of the whole is contained in their intersection.

*Exactness.* Fix T and let X be a random variable with law π, independent of (φ\_t). Each φ\_t preserves π, so Φ(−T → 0)(X) \~ π. On the event {T\* ≤ T} we have Φ(−T → 0)(X) = Y. Therefore, for any A ⊆ S,

> |P(Y ∈ A) − π(A)| = |P(Y ∈ A) − P(Φ(−T → 0)(X) ∈ A)| ≤ P(T\* > T).

The left-hand side does not depend on T and the right-hand side tends to 0, so the left-hand side is 0. ∎

**Remark 1 — why forward coupling fails.** The forward composition Ψ(0 → T′) = Ψ(T → T′) ∘ Ψ(0 → T) grows on the *outside*, so it never stabilises: even after Ψ(0 → T) is constant, later maps keep moving its value. The forward algorithm returns the state *at* the random coalescence time, and conditioning on "coalescence has just happened" biases which state that is. The standard counterexample is a lazy walk on {0, 1, 2} under a coupling in which trajectories can only merge at a boundary state, so the forward output is never 1, while π puts positive mass there.

**Remark 2 — why early stopping fails.** The proof uses a *deterministic* horizon T and the event {T\* ≤ T}. It gives no control over the law of Y conditioned on a data-dependent stopping rule. Since T\* and Y are dependent in general, aborting slow runs or capping the horizon returns a sample from some other distribution entirely.

Three consequences, which between them account for essentially every CFTP bug:

1. **Randomness on \[−T, 0) must be reused when extending to \[−2T, 0).** The argument requires one fixed realisation of the maps. Regenerating fresh maps each iteration is a different, biased algorithm.
2. **Backward, not forward.** Coupling *to* the future — run all states forward from time 0 until they meet — is biased. Counterexample: a random walk on {0, 1, 2} with a coupling where coalescence only happens at the boundary, so the forward meeting value is never 1.
3. **No early stopping.** T\* is correlated with Y. Aborting a slow run, or capping iterations and returning the current state, reintroduces bias — and the bias is invisible. The API must make this structurally impossible.

## 3. Monotone CFTP: from |S| chains to two

Tracking every starting state is hopeless — 2^(L²) for Ising. If S carries a partial order with a maximum ⊤ and minimum ⊥, and the update is monotone (x ≼ y implies f(x,u) ≼ f(y,u) for every u), then every trajectory is sandwiched between those started at ⊥ and ⊤. The backward composition is constant exactly when those two agree. Two chains instead of 2^(L²).

| Model | Partial order | ⊥ / ⊤ | Monotone when |
| --- | --- | --- | --- |
| Ising, heat-bath Glauber | componentwise on spins | all −1 / all +1 | ferromagnetic J > 0, any field (incl. site-dependent) |
| Random-cluster (FK) | inclusion on edge sets | all closed / all open | q ≥ 1 |
| Hard-core on ℤ² | sublattice-flipped (below) | even sites full / odd sites full | graph is bipartite |

Monotonicity for Ising is essentially the FKG/attractiveness property.

The hard-core case is the interesting one. The componentwise order does **not** work: occupancy is antagonistic between neighbours, so raising a site lowers its neighbours. The fix uses bipartiteness — 2-colour the lattice and define

> σ ≼ τ if and only if σ(v) ≤ τ(v) on even sites and σ(v) ≥ τ(v) on odd sites.

Under this flipped order the single-site heat bath *is* monotone, with ⊥ = all even sites occupied and ⊤ = all odd sites occupied (both legal independent sets). This is exactly why monotone CFTP works for hard-core on ℤ² and fails on, say, a triangular lattice — so make the bipartiteness check explicit in the constructor and route non-bipartite graphs to bounding chains.

### 3.1 Worked example: Ising heat bath on a torus

**Target.** On an L×L torus V with spins σ(v) in {−1, +1},

> π(σ) ∝ exp( β Σ over edges u\~v of σ(u)σ(v)  +  h Σ over v of σ(v) ).

**The update.** Conditioning on everything but site v, the weight is proportional to exp(σ(v)·(βS(v) + h)) where S(v) is the sum of the four neighbouring spins. So the heat-bath conditional is the logistic function

> P(σ(v) = +1 | rest) = p(S(v)) = 1 / (1 + exp(−2(βS(v) + h))).

On ℤ², S(v) takes only the five values −4, −2, 0, 2, 4, so p is a five-entry lookup table. Precomputing it is both faster and — critically — guarantees the ⊥ and ⊤ chains compare against bit-identical thresholds.

**The random map.** The randomness for one step is r = (v, U) with v uniform on V and U uniform on \[0,1\]. The map is: set σ(v) = +1 if U ≤ p(S(v)), else −1; leave every other site alone. This is deterministic given r, which is what lets you apply the same r to two states.

**Monotonicity check.** Suppose σ ≼ τ componentwise. Then S(v) computed in σ is ≤ S(v) computed in τ, and for β ≥ 0 the function p is increasing, so p(S\_σ(v)) ≤ p(S\_τ(v)). With the *same* U, the new spin at v satisfies σ′(v) ≤ τ′(v); all other sites are untouched and still ordered. Hence the update is monotone, with ⊥ = all −1 and ⊤ = all +1. Note where β ≥ 0 entered — this is exactly the step that fails for the antiferromagnet.

**Reference implementation.** The whole of monotone CFTP, with nothing elided:

```python
import numpy as np

class IsingHeatBath:
    def __init__(self, lattice, beta, h=0.0):
        self.lat, self.beta, self.h = lattice, beta, h
        self.p = {S: 1.0 / (1.0 + np.exp(-2 * (beta * S + h)))
                  for S in range(-4, 5, 2)}          # five-entry table

    def randomness(self, t, source, k):
        g = source.generator(t)                      # keyed on ABSOLUTE time t
        return g.integers(0, self.lat.n_sites, k), g.random(k)

    def apply(self, sigma, r):                       # pure: no RNG inside
        sigma = sigma.copy()
        sites, us = r
        for v, u in zip(sites, us):
            S = int(sigma[self.lat.neighbours(v)].sum())
            sigma[v] = 1 if u <= self.p[S] else -1
        return sigma

    def bottom(self): return -np.ones(self.lat.n_sites, dtype=np.int8)
    def top(self):    return  np.ones(self.lat.n_sites, dtype=np.int8)


def monotone_cftp(dyn, source, k=64):
    T = k
    while True:
        lo, hi = dyn.bottom(), dyn.top()
        for t in range(-T, 0, k):                    # blocks of k updates
            r = dyn.randomness(t, source, k)         # same r for both chains
            lo, hi = dyn.apply(lo, r), dyn.apply(hi, r)
        if np.array_equal(lo, hi):
            return lo
        T *= 2
```

The entire correctness argument of §2.1 rests on one line: `dyn.randomness(t, source, k)` is keyed on the absolute time t, so when T doubles, the blocks covering \[−T, 0) regenerate bit-for-bit identically. Change that argument to a loop counter and the sampler is quietly wrong.

**First experiment.** Take a 3×3 torus, enumerate all 512 configurations to get π exactly, draw 10⁴ samples, and run the G-test — then repeat 200 times and KS-test the p-values against Uniform(0,1). Then run the identical harness on `BurnInSampler` with a short burn-in from all-plus and confirm it fails. Separately, record the empirical distribution of T\* on a 4×4 torus at β = 0.3 and at β = 0.6 (the critical value is ln(1+√2)/2 ≈ 0.4407); the gap between those two histograms is your first real result and belongs in the README.

## 4. Bounding chains for the non-monotone case

When no useful order exists, replace "track two extremal states" with "track a set". A bounding chain holds, at each site, a *subset* of the local state space, with the invariant: if a state x lies inside the bound, then f(x, u) lies inside the updated bound. Coalescence is certified when every site's set is a singleton.

For hard-core the local states are {0, 1, ?}. Heat-bath update at site v with uniform U:

1. If U > λ/(1+λ): set the site to 0 (unconditionally unoccupied).
2. Else if **all** neighbours are certainly 0: set it to 1.
3. Else if **some** neighbour is certainly 1: set it to 0.
4. Else: set it to ?.

Two things matter. The bounding chain is a *sufficient* certificate — it can miss coalescence that really happened, so it is conservative but never wrong. And Huber proves polynomial coalescence for λ < 2/(Δ−2), which gives a concrete regime to benchmark in. The same machinery gives perfect samples of proper q-colourings for q large relative to Δ (Huber's original bound q > 3Δ; later work improves it).

The property-based test writes itself: draw a random state inside the bound, evolve state and bound under the *same* randomness, assert containment is preserved. If containment ever breaks, either the code or the bounding chain is wrong. This one test is worth more than fifty unit tests.

## 5. Read-once CFTP (Wilson 2000)

Doubling CFTP needs the randomness on \[−T, 0) available repeatedly. Wilson's restructuring removes that need entirely: every random number is consumed exactly once, memory is O(1), and it works off a streaming entropy source.

Fix a block length k, let B be the composite map of k steps, and let p = P(B is constant) > 0. Index blocks backwards from time 0 and let T be the first index with B(−T) constant. The CFTP output is B(−1) ∘ … ∘ B(−(T−1)) applied to the value of B(−T). Now note three facts:

- T − 1 is Geometric(p);
- B(−1), …, B(−(T−1)) are i.i.d. distributed as (B given not constant);
- B(−T) is distributed as (B given constant), independent of them.

All three are reproduced by generating blocks **forwards**:

```
x = None
loop forever:
    C = next_block()                  # fresh randomness, never stored
    if C.is_coalescent:
        if x is None:  x = C.value    # plays the role of B(-T)
        else:          return x       # the stopping signal
    else:
        if x is not None:  x = C.apply(x)
```

The first phase is rejection sampling for a draw from (B given constant). The run of non-coalescent blocks before the *next* constant block is Geometric(p) and i.i.d. conditioned non-constant. Expected number of blocks is 2/p, so choose k to make p about 1/2 — via a pilot calibration whose randomness is then discarded.

## 6. Verification is the real product

On a small graph you can compute π exactly by enumerating all 2^n configurations and summing the Boltzmann weights. That lets you ask the question no burn-in scheme can answer: is the sampler's output distribution equal to the target? Design it properly — a single chi-squared on one run proves very little.

- **Test statistics.** Full multinomial G-test over the state space for tiny lattices (a 3×3 torus is 512 states); binned sufficient statistics (energy, magnetisation, nearest-neighbour correlation) for larger lattices, which is more powerful per sample and more interpretable.
- **Meta-test.** Run the whole experiment R times independently, collect R p-values, KS-test them against Uniform(0,1). This catches small biases a single test misses, and it is the honest way to make the claim.
- **Positive controls.** Ship the deliberately broken samplers — short burn-in, coupling-to-the-future, regenerated-randomness "CFTP", truncated CFTP — and demonstrate the test rejects them. A validation suite with no demonstrated power is decoration. This is the most persuasive artifact in the project.
- **Beyond enumeration.** For tori too big to enumerate, get exact reference values from the transfer matrix (a 2^L × 2^L matrix, fine to L ≈ 12) and from the Ferdinand–Fisher closed form for the finite-torus free energy. That allows validation at L = 8, where brute force is dead.

## 7. Package layout

Working name `perfectsim` — check PyPI before committing to it.

```
perfectsim/
  rng/          counter.py  stream.py
  core/         protocols.py  cftp.py  readonce.py  bounding.py
                schedules.py  result.py
  models/       ising.py  random_cluster.py  hardcore.py  colourings.py
  lattices/     graph.py  grid.py
  validation/   enumerate.py  transfer_matrix.py  gof.py
                baselines.py  report.py
  backends/     numpy_kernels.py  numba_kernels.py
  cli.py
```

Core depends on numpy alone; numba, scipy and networkx are optional extras. A library whose core has one dependency is a library people actually install.

## 8. Core protocols and the randomness source

The decisive design decision: **separate drawing randomness from applying it**, so one randomness object can be applied to several states. Everything else follows.

```python
S = TypeVar("S")   # state
R = TypeVar("R")   # a block of randomness

class Dynamics(Protocol[S, R]):
    def randomness(self, t: int, source: TimeIndexedRandomness) -> R:
        """Randomness for absolute time-block t. Deterministic in (seed, t)."""
    def apply(self, state: S, r: R) -> S:
        """Pure and deterministic. MUST NOT consume any RNG."""
    def copy(self, state: S) -> S: ...

class MonotoneDynamics(Dynamics[S, R], Protocol):
    def bottom(self) -> S: ...
    def top(self) -> S: ...
    def leq(self, x: S, y: S) -> bool: ...      # tests and assertions only
    def equal(self, x: S, y: S) -> bool: ...

class BoundingDynamics(Protocol[B, R]):
    def initial_bound(self) -> B: ...           # maximal uncertainty
    def apply(self, bound: B, r: R) -> B: ...
    def is_singleton(self, bound: B) -> bool: ...
    def collapse(self, bound: B) -> S: ...
    def contains(self, bound: B, state: S) -> bool: ...   # tests
```

If `apply` is pure and takes its randomness as an argument, it is impossible to accidentally hand the top and bottom chains different random numbers. The type system does correctness work for you — say so in the docs.

```python
class TimeIndexedRandomness:
    """Counter-based (Philox) source: block t is reproducible from (seed, t)
    in O(1) time and O(1) memory. Random numbers are never stored."""
    def __init__(self, seed: int, bit_generator: str = "Philox"): ...
    def generator(self, t: int) -> np.random.Generator: ...
    def uniforms(self, t: int, n: int) -> NDArray[np.float64]: ...
    def integers(self, t: int, n: int, high: int) -> NDArray[np.int64]: ...
```

This is the neatest part of the design and deserves its own docs page. Using `np.random.Philox(key=seed, counter=t)` turns "reuse the same randomness when extending the horizon" from a discipline you must maintain into an automatic consequence of indexing by absolute time. Doubling from T to 2T costs no memory at all. Benchmark it against store-the-uniforms and store-the-seeds, and put the table in the README.

### 8.1 How the randomness actually works

An ordinary generator holds internal state and advances it on every draw: to reach the 1000th number you must generate the 999 before it. Reusing randomness then means either storing it or re-drawing from the start in exactly the same order — so correctness depends on every round consuming numbers in an identical sequence, and one stray draw shifts the alignment silently.

Philox is a counter-based generator, which is closer to a block cipher than to a sequence. It computes

> output = F(key, counter)

directly, with no internal state to advance. Counter 10⁶ costs exactly what counter 1 costs. Think of it as an astronomically large fixed table of random-looking numbers: the key picks which table, the counter picks the row.

```python
class TimeIndexedRandomness:
    def __init__(self, seed):
        self.seed = seed

    def generator(self, t):
        bg = np.random.Philox(key=self.seed, counter=t)
        return np.random.Generator(bg)
```

The counter is the **absolute time index**, so `generator(-256)` returns the stream for time −256 forever — first round or fifth, nothing stored, no order dependence between calls. In practice use one counter per *block* of k steps rather than per step, since constructing a generator has overhead; the block then draws its k sites and k uniforms from that one generator.

**The two independence axes.** Randomness must be reproducible along one and independent along the other, and a single mechanism covers both without the two ever conflicting:

|  | Same run | Different runs |
| --- | --- | --- |
| **Same time index** | identical — what the theorem needs | independent — what makes the samples i.i.d. |

The key handles the columns, the counter handles the rows. Each run's key is fixed when its job is created and travels with it through every re-queue at a doubled horizon.

**The mental model.** Each run owns an infinite sequence u(−1), u(−2), u(−3), … fixed the moment its key is chosen. Increasing the horizon generates nothing new — it looks further back into a sequence already determined. Philox makes "already determined" literal: you can query position −10⁶ without having touched positions −1 through −999,999.

**Caveats to document.** Derive per-run keys with `SeedSequence.spawn` rather than picking them by hand, since seeds differing only slightly can give correlated streams in some generators. Pin `Philox` explicitly rather than relying on the numpy default, and treat any change to how many draws a block makes as a change that invalidates stored golden outputs.

## 9. Algorithms, schedules and results

```python
# core/cftp.py
def monotone_cftp(dyn: MonotoneDynamics, source: TimeIndexedRandomness, *,
                  schedule: Schedule = DoublingSchedule(start=16),
                  max_steps: int | None = None,
                  record: RunRecorder | None = None) -> CFTPResult[S]: ...

def bounding_chain_cftp(dyn: BoundingDynamics, source, *, schedule=...) -> CFTPResult[S]: ...

# core/readonce.py
def read_once_cftp(dyn: MonotoneDynamics, source, *, block_length: int) -> CFTPResult[S]: ...
def calibrate_block_length(dyn, source, *, target_p: float = 0.5,
                           pilot_runs: int = 30) -> Calibration: ...

# core/schedules.py
class Schedule(Protocol):
    def horizons(self) -> Iterator[int]: ...
class DoublingSchedule(Schedule): ...   # 16, 32, 64 — within a factor 2 of optimal work
class LinearSchedule(Schedule): ...     # teaching and comparison only
```

```python
@dataclass(frozen=True, slots=True)
class CFTPResult(Generic[S]):
    sample: S
    horizon: int              # final T
    total_updates: int        # work done, including discarded passes
    n_extensions: int
    wall_time: float
    provenance: Provenance    # seed, model params, version, git sha
```

**The API rule:** `max_steps` raises `CoalescenceTimeout`. It never returns a state. If someone genuinely wants a truncated run for exploration, they get it from a separately named function returning a `BiasedSample` type that is not assignable to `Sample`. Make bias a type error.

### 9.1 Running many samples: vectorisation and batching

Numpy pays its Python overhead once per *call*, not per element, so the whole game is finding large collections of things to do in one call. Time is not one of them: σ(t+1) reads what σ(t) wrote, so the CFTP inner loop is irreducibly sequential. Parallelism has to come from elsewhere, and there are exactly two places it can come from.

| Axis | What it exploits | Good when |
| --- | --- | --- |
| Across sites | conditional independence of a sublattice (checkerboard) | large lattice, few samples |
| Across runs | separate CFTP executions share nothing at all | small lattice, very many samples |

**B is the number of independent CFTP runs executed simultaneously** — a tuning knob, typically 500–2000, chosen so each numpy call has enough elements to be worth making. They are not one chain's successive states: each run produces one exact draw, with its own lattice, its own key and its own coalescence time.

**Layout.** Hold the state as `(2B, n)` with n = L² flat sites: rows 0…B−1 are the ⊥ chains, rows B…2B−1 the ⊤ chains. The randomness for one step is two length-B arrays (the site each run updates, the uniform each run draws), tiled so that row b and row B+b receive identical values. Two benefits: double the vectorisation width, and the invariant that the sandwiching chains see the same randomness becomes a property of the layout rather than a rule to remember. A precomputed neighbour table `nbr` of shape `(n, 4)` encodes the periodic boundary once, so nothing downstream handles wraparound.

```python
rows = np.arange(2 * B)
S = sigma[rows[:, None], nbr[v]].sum(axis=1)   # (2B,) neighbour sums
p = table[(S + 4) // 2]                        # (2B,) five-entry lookup
sigma[rows, v] = np.where(u <= p, 1, -1)       # write one site per run
```

Since S is an integer, two runs with the same neighbour sum compare against a bit-identical threshold by construction — the floating-point trap of §14 disappears rather than needing discipline.

**Retirement: use a job queue, not a shrinking array.** Runs finish at different times and T\* has a heavy tail, so masking finished rows in a fixed array leaves you running two survivors at the cost of two thousand. Exploit the fact that a doubled round restarts from ⊥ and ⊤ anyway: a job carries no state, only `(key, horizon)`. So keep a queue, run all jobs at a given horizon as one full-width batch, re-queue the failures at 2T, and top up with fresh keys to hold the width near B. Log total updates across all rounds — the final horizon alone understates the work, which is about 2× the final horizon per run.

The rule that keeps this correct: **key each run's randomness on its key, never on its row index.** Rows are reused by other jobs as runs retire, so a row-derived stream would hand one run another's randomness — the reuse bug of §14, arriving through a back door you built yourself.

**Partial batches are forbidden.** The proof in §2.1 fixes a *deterministic* horizon and bounds the error by P(T\* > T); it says nothing about the law of Y conditioned on a run having finished quickly. T\* and Y are dependent — on Ising, slow coalescence goes with the randomness that keeps ⊥ and ⊤ apart, near the bottleneck between the magnetisation modes — so keeping the finished runs and dropping the rest samples from π conditioned on being easy to reach. So `sample_many` returns exactly n samples or raises: no `timeout` argument that quietly returns fewer, no progress-bar cancel handing back the partial batch.

**Why this earns its keep.** Not production sampling — the validation suite. 10⁴ samples × 200 replications on a 3×3 torus is 2 million complete CFTP runs: an overnight job sequentially, a few minutes batched. That is the difference between evidence you generate once and evidence that runs in CI on every commit.

## 10. Models and lattices

```python
class IsingHeatBath(MonotoneDynamics):
    def __init__(self, lattice: Lattice, beta: float,
                 field: float | NDArray = 0.0,
                 scan: Literal["random", "systematic", "checkerboard"] = "checkerboard"): ...
    # + energy(), magnetisation(), correlation(r)

class RandomCluster(MonotoneDynamics):        # single-bond heat bath, q >= 1
    def __init__(self, lattice, p: float, q: float): ...

def potts_from_random_cluster(omega, q, rng) -> NDArray   # colour clusters i.i.d.
def ising_from_random_cluster(omega, rng) -> NDArray

class HardCoreBipartite(MonotoneDynamics):    # sublattice-flipped order
class HardCoreBounding(BoundingDynamics):     # Huber, any graph
class ProperColouringBounding(BoundingDynamics)   # optional
```

The random-cluster module is what earns the library serious credibility: it yields **perfect samples of the q-state Potts model at any temperature, including criticality**, which nothing else in Python does. Its cost is the connectivity query "are u and v joined in ω without edge e?". Start with a local BFS bounded to the component, measure, and only then consider a dynamic-connectivity structure (Euler-tour trees). Be honest about the complexity in the docs rather than hiding it.

```python
@dataclass(frozen=True)
class Lattice:
    n_sites: int
    indptr: NDArray[np.int32]     # CSR neighbour structure
    indices: NDArray[np.int32]
    is_bipartite: bool
    colour: NDArray[np.int8] | None   # 2-colouring for checkerboard sweeps
    @classmethod
    def torus(cls, L: int, M: int | None = None) -> Lattice
    @classmethod
    def grid(cls, L, M, boundary: Literal["free","periodic","fixed"]) -> Lattice
    @classmethod
    def from_edges(cls, n, edges) -> Lattice
```

CSR neighbour arrays rather than a dict of lists: numba-friendly, cache-friendly, and they let the whole hot loop live inside one `@njit` function.

## 11. Validation module, and everything else that ships

```python
def enumerate_distribution(model, *, max_sites: int = 22) -> ExactDistribution
def ising_transfer_matrix(L: int, beta: float, field: float = 0.0) -> ExactSummary
def gof_test(samples, exact, statistic="state"|"energy"|"magnetisation") -> TestResult
def pvalue_uniformity(results: Sequence[TestResult]) -> TestResult   # KS on p-values
def tv_distance(empirical, exact) -> tuple[float, tuple[float, float]]

# validation/baselines.py — the bug museum
class BurnInSampler(Sampler)            # Glauber + fixed burn-in from a fixed start
class ForwardCouplingSampler(Sampler)   # coupling to the future
class ResampledRandomnessCFTP(Sampler)  # the classic reuse bug
class TruncatedCFTP(Sampler)            # gives up after N, returns the state

def validation_report(models, samplers, n_samples, replications) -> Report
```

Beyond the library core:

- **CLI.** `perfectsim sample ising --L 32 --beta 0.44 --n 1000 --out s.npz`, `perfectsim validate ising --L 3 --report out.md`, `perfectsim bench`.
- **Batch sampling.** `sample_many(model, n, seed)` using `SeedSequence.spawn`, parallel across processes. Document the trap: you must keep every sample, including the slow ones. Discarding stragglers to hit a time budget is exactly the bias CFTP exists to remove.
- **Diagnostics worth having.** Empirical distribution of the coalescence time against β, showing the blow-up at criticality for Glauber and the contrast with random-cluster dynamics. Two figures, one story, very quotable.
- **Provenance.** Seed, model parameters, library version and git sha serialised alongside the samples.
- **Docs.** mkdocs-material with mkdocstrings, tutorial notebooks via myst-nb, a benchmarks page, and a "how to add your own model" guide built around the protocols.

## 12. Build plan

Tooling: `uv`, `ruff`, `mypy --strict`, `pytest` with `hypothesis` and `pytest-benchmark`, `pre-commit`, GitHub Actions across Python 3.11–3.13 and three operating systems, Zenodo DOI, `CITATION.cff`, MIT licence.

| Phase | Content | Effort |
| --- | --- | --- |
| 0 | Repo, CI, docs skeleton, empty protocols | 2–3 evenings |
| 1 | Pure-Python Ising heat bath, time-indexed RNG, monotone CFTP with doubling, enumeration tests on 3×3 and 4×4 tori, **and the bug museum** | 1–2 weeks |
| 2 | Performance: checkerboard updating, batching across runs (§9.1), numba kernels, lookup tables, memory benchmarks | 1–2 weeks |
| 3 | Random-cluster and Potts, connectivity queries, validation of q=3 by enumeration | 1–2 weeks |
| 4 | Bounding chains: hard-core on arbitrary graphs, hypothesis containment tests, Huber regime | 1–2 weeks |
| 5 | Read-once CFTP and block calibration | a few days |
| 6 | Validation report, benchmarks, figures, JOSS submission | 1 week |

Notes on the phases that have subtleties:

- **Phase 1 ships as v0.1.0 and is already a respectable portfolio piece.** Write the broken samplers here rather than later: building the tests against known-wrong implementations is how you find out whether your tests have power.
- **Checkerboard updating (phase 2).** On a bipartite lattice all even sites are conditionally independent given the odd sites, so a half-sweep is one vectorised numpy operation. Large speedup, still monotone, still exact — but it is a *different chain* from single-site Glauber, so document it as such and report its coalescence times separately.
- **Lookup tables (phase 2).** Precompute the heat-bath acceptance probability indexed by the neighbour sum; only 2Δ+1 distinct local fields exist. Faster, and bitwise identical across the ⊥ and ⊤ chains, which matters (see traps).
- **Phase 4 cross-check.** Hard-core via bounding chains and via the bipartite monotone order should agree in distribution on ℤ². Another free validation.
- **Phase 6.** The Journal of Open Source Software reviews research software with proper tests and docs publicly and issues a citable DOI. That turns "GitHub repo" into "publication".

## 13. Reading list, in reading order

1. **Häggström, *Finite Markov Chains and Algorithmic Applications* (2002), ch. 10–12.** The cleanest elementary account, with the forward-versus-backward and reuse-randomness pitfalls spelled out. One evening. Start here.
2. **Propp & Wilson (1996), "Exact sampling with coupled Markov chains and applications to statistical mechanics", RSA 9:223–252.** The source. Read for the monotone construction and the doubling argument.
3. **Huber, *Perfect Simulation* (CRC Press, 2016).** The reference text for the whole field: CFTP, bounding chains, read-once, dominated CFTP, randomness recycler. If you buy one book, this one.
4. **Wilson (2000), "How to couple from the past using a read-once source of randomness", RSA 16:85–113.**
5. **Huber (2004), "Perfect sampling using bounding chains", Ann. Appl. Prob. 14:734–753**, plus Huber (1998, STOC) for colourings.
6. **Levin & Peres, *Markov Chains and Mixing Times* (2nd ed.), ch. 25.** CFTP in mixing-time language, which is what you want for the complexity discussion.
7. **Grimmett, *The Random-Cluster Model*.** FK monotonicity, the coupling with Potts, and the q ≥ 1 conditions.
8. **Fill (1998)**, interruptible perfect sampling, and Fill–Machida–Murdoch–Rosenthal (2000). Fixes the impatient-user bias; worth a docs page even if unimplemented.
9. **Kendall & Møller (2000)**, dominated CFTP — needed if you extend to point processes.
10. **Guo, Jerrum & Liu (2019), "Uniform sampling through the Lovász Local Lemma", JACM.** Partial rejection sampling: a genuinely different modern paradigm and an excellent comparison baseline for hard-core.
11. **Salmon et al. (2011), "Parallel random numbers: as easy as 1, 2, 3".** Philox and Threefry, the basis of the RNG design.
12. **David Wilson's exact-sampling bibliography**, long the most comprehensive single source on perfect simulation. Worth mining for further models.

## 14. Traps

| Trap | Defence |
| --- | --- |
| Regenerated randomness on extension | Counter-based RNG indexed by absolute time. Test that `source.uniforms(t, n)` is bit-identical across calls and processes. |
| Composition order | New randomness at *more negative* times is applied first. Extending T → 2T, the new segment covers \[−2T, −T) and the old covers \[−T, 0). |
| Caching the state at time −T | You cannot. After extending, the chains entering −T are no longer ⊥ and ⊤, so the whole \[−T, 0) segment must be replayed. (Caching the composite map is fine when the state space is tiny — teaching examples only.) |
| Scan order treated as separate randomness | With random scan, the site sequence is part of the randomness and must come from the same time-indexed block. |
| Floating-point non-determinism | Compute the acceptance threshold identically for both chains via lookup tables, not by recomputing from possibly differently ordered sums. One ULP at u ≈ p breaks coalescence detection silently. |
| Global RNG state | No `random`, no `np.random.seed`, ever. |
| Assumed monotonicity | Hypothesis test: draw x ≼ y and random r, assert `leq(apply(x,r), apply(y,r))`. This is what catches the hard-core sublattice sign error, and it will catch it. |
| Flaky CI | Fixed seeds in the main suite so failures reproduce; a nightly job with random seeds and Bonferroni-corrected alarms to catch seed-specific luck. |
| Randomness keyed on a batch row index | Key it on the run's own Philox key, which travels with the job. Rows get reused by other jobs as runs retire (§9.1), so a row-derived stream hands one run another's randomness. |
| Returning a partial batch | sample\_many returns exactly n samples or raises. Dropping slow runs conditions on {T\* <= T}, which the proof of 2.1 explicitly does not cover. |

One non-trap worth knowing: **early coalescence is fine.** Once ⊥ and ⊤ meet mid-run they stay met, so you may collapse to a single chain and continue. That is an optimisation, not a shortcut past time 0.

## 15. Scope, positioning, and what comes next

**Scope discipline.** The minimum shippable library is phase 1 alone: monotone CFTP, Ising, exact validation, bug museum. Release it. Everything after is an independently releasable layer. The failure mode here is six models built badly instead of two built well.

**Positioning.** The prior art is thin and mostly in R — CRAN's [IsingSampler](https://search.r-project.org/CRAN/refmans/IsingSampler/html/IsingSampler.html) offers CFTP as one of three sampling methods, and [ROCFTP.MMS](https://nabipoor.r-universe.dev/ROCFTP.MMS/doc/manual.html) implements read-once CFTP with a Metropolis-multishift coupler for univariate posteriors. Python has notebook-grade implementations. State that gap precisely in the README's first paragraph, then state the differentiators: model-agnostic protocols, bounding chains, read-once, counter-based randomness, and a validation suite with demonstrated power. Vague novelty claims read as naivety; specific ones read as scholarship.

**Complexity honesty.** CFTP's expected running time relates to the mixing time, but the tail can be heavy, and for Glauber Ising below the critical temperature it is exponential in L. The library should *demonstrate* this rather than hide it — the coalescence-time-versus-β plot and the contrast with random-cluster dynamics is the most scientifically interesting figure in the project. A library that tells the truth about when it is slow is more trustworthy than one that does not.

**The natural sequel.** Perfect simulation is not only a discrete-MCMC topic. The Exact Algorithm of Beskos & Roberts (2005) simulates diffusion sample paths from the law of the SDE with no discretisation error, via retrospective rejection sampling against a Brownian proposal using Girsanov and a Poisson thinning argument (EA1/EA2/EA3, then ε-strong simulation in Beskos–Peluchetti–Roberts 2012). It is the continuous-time analogue of the same idea: exactness through clever coupling rather than through taking a limit. A `perfectsim.diffusions` module would make the library unusually coherent. Keep it firmly out of v1, but know the destination while designing the abstractions.

**Application angle.** The project touches monotone coupling, FKG, mixing times, the random-cluster representation and LLL-based sampling — all live areas with active groups. A tested, documented library plus a JOSS paper is a far stronger signal than a notebook, and it gives you something concrete to open an email to a prospective supervisor with.
