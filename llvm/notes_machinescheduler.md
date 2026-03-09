# GenericScheduler (Pre‑RA Machine Instruction Scheduler)

[`llvm/lib/CodeGen/MachineScheduler.cpp`](https://github.com/mahesh-attarde/llvm-project/blob/walk/llvm/lib/CodeGen/MachineScheduler.cpp)  
 `GenericScheduler` is the *default* pre‑register‑allocation (pre‑RA) bidirectional instruction scheduler in LLVM’s MachineScheduler framework.

## 1. Purpose & Design Philosophy

`GenericScheduler` balances (in a *layered / hierarchical heuristic*):

| Objective | Why It Matters Pre‑RA |
|-----------|-----------------------|
| Register Pressure | Avoid spills: cheaper than fixing after RA |
| Critical Path Latency | Reduce execution time in latency-bound code |
| Functional / Proc Resource Balance | Prevent bottlenecks (e.g. issue width, ports) |
| Cluster Preservation | Enable later macro-fusion, load grouping, post-RA pairing |
| Physical Register Adjacency | Tighten physreg live ranges (e.g. implicit defs) |
| Hazard & Stall Avoidance | Avoid unbuffered pipeline stalls |
| Determinism | Stable fallback ordering (`NodeOrder`) |

**Key Attributes:**
- **Bidirectional scheduling** (Top & Bottom frontiers converge)
- **Pressure-aware** (if region size justifies cost)
- **Resource-aware** (uses `TargetSchedModel`)
- **Latency-aware** (depth/height path metrics + cyclic critical path logic)
- **Extensible** (pluggable policies / DFS subtree integration)
- **Explainable** (each decision tagged with a `CandReason` statistic)

## 2. Components API

| Component | Role |
|-----------|------|
| `ScheduleDAGMILive` | DAG + LiveIntervals + pressure deltas |
| `SchedBoundary` (Top & Bot) | Per-direction frontier: ready queues, cycle, resources |
| `SchedRemainder Rem` | Tracks global remaining resource/latency summary |
| `SchedCandidate` | Structure under comparison for selection |
| `CandPolicy` | Per-pick policy (latency vs resource emphasis) |
| `RegPressureTracker` | Computes deltas & max set pressure |
| `SchedModel` | Target micro-architectural description |
| `ClusterInfo` | Groups of SUnits (load/store or mutation-defined clusters) |

## 3. Lifecycle 
```
initialize() →
  compute (optional) DFS result for ILP / subtree awareness
  init Top & Bot boundaries (SchedBoundary)
  build Remaining resource baseline (Rem.init)

initPolicy() →
  decide directionality & pressure tracking (RegionPolicy)
  apply subtarget overrides + CLI flags

registerRoots() →
  compute critical path (ExitSU depth + check other roots)
  optionally compute cyclic path (loop self-edge analysis)

schedule loop →
  while (Top != Bottom):
     SU = pickNode()
     schedNode(SU, isTop)
     DAG updates + release successors/preds
```

## 4. Region Policy Initialization

```c++
void GenericScheduler::initPolicy(..., unsigned NumRegionInstrs) {
  // Heuristic: only track pressure if region large enough relative to avail regs
  RegionPolicy.ShouldTrackPressure = (NumRegionInstrs > NIntRegs/2);
  RegionPolicy.OnlyBottomUp = true; // default bias
  MF.getSubtarget().overrideSchedPolicy(RegionPolicy, Region); // target hook
  // CLI overrides: -misched-prera-direction
  if (!EnableRegPressure) RegionPolicy.ShouldTrackPressure = false;
}
```
**Why conditional pressure tracking?**  
Pressure tracking can be expensive; small regions rarely spill. A region-size heuristic avoids needless overhead.

## 5. Internal State Objects

### 5.1 `SchedRemainder Rem`
Tracks “what’s left” globally:
```c++
struct SchedRemainder {
  unsigned RemIssueCount;            // Remaining (scaled) micro-ops
  SmallVector<unsigned> RemainingCounts; // Per proc resource
  unsigned CriticalPath;             // Acyclic path length (depth of ExitSU)
  unsigned CyclicCritPath;           // Loop-based path (if detected)
  bool IsAcyclicLatencyLimited;      // Derived flag
};
```
### 5.2 `SchedBoundary`
Each of `Top` & `Bot`:
```c++
struct SchedBoundary {
  ReadyQueue Available, Pending;
  unsigned CurrCycle;
  unsigned CurrMOps;        // micro-ops issued this cycle
  unsigned RetiredMOps;     // cumulative issued
  unsigned ExpectedLatency; // scheduled depth frontier (top)
  unsigned DependentLatency;// scheduled height frontier (bottom)
  unsigned ZoneCritResIdx;  // currently critical resource index
  bool IsResourceLimited;   // updated after each schedule
  ...
};
```

### 5.3 `SchedCandidate`
Represents a potential pick:
```c++
struct SchedCandidate {
  SUnit *SU = nullptr;
  bool AtTop = false;
  CandReason Reason = NoCand;
  CandPolicy Policy;
  RegPressureDelta RPDelta;         // Excess, CriticalMax, CurrentMax
  SchedResourceDelta ResDelta;      // CritResources, DemandedResources
  ...
};
```

## 6. Bidirectional Scheduling Mechanics

```
RegionBegin → schedule forward (Top boundary)
RegionEnd   → schedule backward (Bottom boundary)
Converge until CurrentTop == CurrentBottom
```

**High-Level Decision:**
1. Try `pickOnlyChoice()` from `Bot` (if only one candidate)
2. Then `pickOnlyChoice()` from `Top`
3. Else produce `BotCand` & `TopCand` using `pickNodeFromQueue`
4. Compare cross-boundary (reduced heuristic set)
5. Use hierarchical heuristics to pick final SU

## 7. Candidate Initialization & Pressure Deltas
```c++
void GenericScheduler::initCandidate(SchedCandidate &Cand, SUnit *SU, bool AtTop,
                                     const RegPressureTracker &RPTracker,
                                     RegPressureTracker &Temp) {
  Cand.SU = SU;
  Cand.AtTop = AtTop;
  if (DAG->isTrackingPressure()) {
    if (AtTop)
      Temp.getMaxDownwardPressureDelta(..., Cand.RPDelta,...);
    else
      RPTracker.getUpwardPressureDelta(... precomputed PressureDiff ..., Cand.RPDelta,...);
  }
}
```
**Directional nuance:**
- **Top scheduling** = “downward” deltas (what future could inflate)
- **Bottom scheduling** = accurate upward deltas incorporating precomputed per-SU pressure diff (adjusted for lane masks & live-through)

## 8. Policy Computation (`setPolicy`)

```c++
void setPolicy(CandPolicy &P, bool PostRA, SchedBoundary &Zone, SchedBoundary *Other) {
  // Determine if other side or remainder is resource-limited
  // Possibly enable P.ReduceLatency if current cycle > (CritPath - RemainingLatency)
  // If Zone resource-limited && no ReduceResIdx assigned → set it
  // If Other side resource-limited → set DemandResIdx
}
```

**Latency-Limited Detection:**
```c++
RemainingLatency = max( Zone.dependentLatency,
                        max depth among Zone.Available & Pending )
if (RemainingLatency + CurrCycle > CriticalPath)
    P.ReduceLatency = true;
```

**Why two resource knobs?**
- `ReduceResIdx`: actively shrinking local pressure on critical resource
- `DemandResIdx`: opportunistically driving utilization for resources *outside* zone likely to become critical


## 9. Heuristic Layer Ordering

In `tryCandidate` (simplified):

| Priority | Heuristic | Condition |
|----------|-----------|-----------|
| 1 | FirstValid | If no current candidate |
| 2 | PhysReg | `biasPhysReg()` result higher |
| 3 | RegExcess | Fewer / negative pressure set overflows |
| 4 | RegCritical | Less increase on already exceeded critical PSets |
| 5 | (Same boundary) Stall | Lower unbuffered latency stall cycles |
| 6 | Cluster | Maintains cluster succession |
| 7 | (Same boundary) Weak | Fewer weak deps remaining |
| 8 | RegMax | Lower current peak pressure change |
| 9 | ResourceReduce | Lower critical resource increments |
| 10 | ResourceDemand | Higher demand on target resource |
| 11 | (Latency) TopDepth / TopPath / BotHeight / BotPath | Reduce depth/height intelligently |
| 12 | NodeOrder | Earlier original node number (Tie-break) |
| 13 | FirstValid (fallback) | Shouldn’t re-trigger if already assigned |

**Short-Circuit Behavior:** Once a higher-priority heuristic yields strict preference (`TryCand.Reason` updated), remaining heuristics are skipped for that comparison.
## 10. Cluster Awareness

Cluster logic:
```c++
bool isTheSameCluster(CurrClusterID, SU->ParentClusterIdx)
```
- Each pick updates frontier’s `TopClusterID` / `BotClusterID`
- `Cluster` heuristic encourages following already chosen cluster members
- Clusters formed primarily by DAG mutations (e.g., memory op clustering)


## 11. Latency Handling Subtleties

```c++
tryLatency(TryCand, Cand, Zone):
  if (Zone.isTop()):
      // Depth is inverse slack metric at top
      if (max(depths) > scheduledLatency) pick smaller depth
      else tie-break via greater height
  else:
      // symmetric for bottom
```

**Why height at top & depth at bottom secondarily?**  
Encourages central convergence of paths with balanced slack (cutting both long forward and reverse chains).

## 12. Resource Modeling Integration

During `SchedBoundary::bumpNode(SU)`:
- Compute per-resource reservations via `countResource`
- Update `ExecutedResCounts`
- Possibly change `ZoneCritResIdx` if usage crosses micro-op scaled threshold
- Detect stalls: if *reserved-until* > `CurrCycle`, cycle is advanced (`bumpCycle`)
- After schedule:
  ```c++
  IsResourceLimited = checkResourceLimit(LatFactor,
                                         getCriticalCount(),
                                         getScheduledLatency(), /*After*/ true);
  ```

**Dynamic switching:** If micro-op retirement overtakes resource usage cycles → critical resource resets to “issue width” (meaning instruction dispatch throughput becomes limit).

## 13. Physical Register Bias & Rescheduling

```c++
int biasPhysReg(SU, isTop):
  if copy:
     prefer to place copy near its physreg user/def
  if move-immediate to phys regs only:
     delay at top (return -1) to keep imm-lifetime short
```

**After scheduling an SU with physreg defs/uses:**
```c++
reschedulePhysReg(SU, isTop):
  scan its (preds/succs) single-user physreg copies
  move those copies immediately adjacent to SU
```

Goal: shrink live range of physregs, aiding register assignment & avoiding stalls around special implicit registers (e.g. flags on some targets, accumulator pairs, special mul/div result registers).


## 14. DFS Result Integration (Optional)

If `RegionPolicy.ComputeDFSResult` is set:
```c++
DAG->computeDFSResult();
DFSResult provides:
  - Subtree IDs
  - ILP metrics
  - Subtree levels
Used implicitly via alternative strategies or for advanced heuristics/experiments.
```
`GenericScheduler` itself does **not** heavily rely on ILP ranking—the baseline variant is latency/resource/pressure centric. ILP schedulers are separate strategies (`ilpmax`, `ilpmin`) registered via `MachineSchedRegistry`.


## 15. Statistics & Diagnostics

Each pick increments counters:
```c++
NumTopPreRA / NumBotPreRA
NumRegExcessPreRA, NumStallPreRA, ...
NumClusterPreRA, NumWeakPreRA, ...
NumNodeOrderPreRA, NumFirstValidPreRA
```
Diagnostic macro:
```c++
tracePick(Cand.Reason, IsTopNode)
```

Enables aggregate reasoning about which heuristics dominate scheduling choices in a workload.


## 16. Failure Modes & Defensive Checks

| Potential Issue | Preventative Logic |
|-----------------|--------------------|
| Node released twice | `assert(SU->NumPredsLeft > 0)` in release functions |
| Permanent hazard | Cycle bump loop (with assert disabled due to PR note) |
| Queue overflow cost | `ReadyListLimit` early gating |
| Invalid resource overlap | Assertions in `ResourceSegments::intersects` |
| Dangling iterators after moves | Region iterators re-derived post-schedule |
| SU reused after scheduling | `isScheduled` flag + ready removal |


## 17. Comparison vs PostGenericScheduler

| Aspect | GenericScheduler (Pre‑RA) | PostGenericScheduler (Post‑RA) |
|--------|---------------------------|--------------------------------|
| Register Pressure | Enabled (unless small region) | Generally disabled |
| Lane Masks | Possible | Usually unused |
| Heuristic depth | Full stack | Reduced (no pressure tiers) |
| Physreg bias | Important (pre RA copies) | Less pronounced |
| Latency emphasis | Balanced with pressure | More direct hazard & resource focus |
| DFS use | Optional | Rarely central |
| Mutations benefit | CopyConstrain helps RA | Primarily clustering/fusion |


## 18. Pseudocode of Core Picking Logic

```c++
while (!regionFinished) {
  if (RegionPolicy.OnlyTopDown)
      SU = pickFromTop();
  else if (RegionPolicy.OnlyBottomUp)
      SU = pickFromBottom();
  else
      SU = pickBidirectional(); // produce TopCand & BotCand

  scheduleNode(SU, isTopSide);
  releaseDependencies(SU);
}
```

`pickBidirectional()` pseudo-simplified:
```c++
if (SU = Bot.pickOnlyChoice()) return SU;
if (SU = Top.pickOnlyChoice()) return SU;

BotPolicy = derive(Bot, Top);
TopPolicy = derive(Top, Bot);

BotCand = bestCandidate(Bot);
TopCand = bestCandidate(Top);

return crossCompare(BotCand, TopCand);
```


## 19. Extending / Modifying GenericScheduler

| Goal | Extension Point |
|------|------------------|
| Add new heuristic dimension | Modify `tryCandidate` layering |
| Replace heuristic ordering | Rework hierarchy into unified cost model |
| Add mutation behavior | Create new `ScheduleDAGMutation` and register in DAG builder |
| Integrate ML ranking | Wrap `pickNodeFromQueue` to pre-rank SU list |
| Custom subtarget policy | Implement `overrideSchedPolicy()` in subtarget |


## 20. Gaurdrails

| If You Observe… | Consider |
|------------------|----------|
| Excess spills | Check `RegExcess` counters & enable pressure tracking |
| Cluster not respected | Ensure mutation ran (`-misched-cluster`) |
| Long stalls | Look at `Stall` picks; maybe target model lacks unbuffered resource flags |
| Backward scheduling dominates | Direction overrides (`-misched-prera-direction=topdown`) |
| Too slow scheduling | Disable pressure (`-misched-regpressure=false`) to verify impact |
| Unstable order noise | Inspect `NodeOrder` picks; adjust heuristics or tie-break |


## 21. Modelling
Think of `GenericScheduler` as **two hands knitting the instruction stream inward**:
- Each hand sees local ready nodes, annotated with multi-dimensional pressures, latencies, and resource implications.
- It decides which stitch (instruction) to lay next by applying a fixed *preference ladder* instead of a monolithic score.
- DAG mutations influence adjacency via soft (weak / cluster) edges.
- Register pressure expresses “oxygen budget” needed to avoid spills.


## 22. Summary Table

| Category | Highlight |
|----------|-----------|
| Core Strength | Balanced multi-factor scheduling with interpretability |
| Design Style | Hierarchical heuristics (short-circuit) |
| State Separation | Boundary (issuer state) vs Remainder (global future) |
| Complexity Control | Ready list cap, conditional pressure |
| Target Integration | Resource model + hazard recognizer + policy override |
| Major Risks Avoided | Spill cascades, unused critical path slack, port oversubscription |
| Expandability | Strategy registry + DAG mutations |

## 24. Key Source Anchors (Search Terms)

| Concept | Identifier to grep |
|---------|--------------------|
| Candidate evaluation | `tryCandidate` |
| Pressure delta setup | `initCandidate` |
| Scheduling loop entry | `pickNode` |
| Direction arbitration | `pickNodeBidirectional` |
| Resource accounting | `countResource` / `bumpNode` |
| Latency detection | `shouldReduceLatency` |
| Cluster follow logic | `TopClusterID` / `BotClusterID` |
| Copy physreg tightening | `reschedulePhysReg` |
| Policy derivation | `setPolicy` |

## TODO:
-  `SchedBoundary::bumpNode` to internalize cycle/resource transitions.
- `-misched=ilpmax` to see different tradeoff spaces.
- cluster effects by enabling dag viewing: `-view-misched-dags`.

