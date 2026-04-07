# Transformer Agent Microkernel Roadmap

## Title

Future-possible roadmap for a transformer-native agent control plane

## Status

- Date: April 6, 2026
- Horizon: 12 months of focused exploratory engineering
- Status: speculative and additive
- Relationship to current work: complements `docs/roadmaps/current-roadmap.md`

## Product statement

Build a deterministic transformer-native microkernel that supervises long-horizon agent execution while treating probabilistic LLM reasoning as a controlled external service.

The target is not "run Linux inside the transformer" and not "fully port `agent-os` into attention."

The target is narrower and more credible:

- keep a thin host for real time, persistence, network, filesystem backing, and sandboxed effects,
- move scheduling, queues, timers, message passing, capability checks, session state, and tool/LLM orchestration into a deterministic transformer-shaped kernel,
- let probabilistic LLM calls happen through explicit typed syscalls such as `CALL_MODEL`,
- make the resulting system accelerator-native, batchable, resumable, and auditable.

## Why this exists

The current agent tooling trend is:

- an LLM runs inference on GPU,
- a CPU harness handles queues, timers, retries, tools, filesystem state, and scheduling,
- long-horizon behavior emerges from repeated host-mediated tool loops.

That pattern works, but it has clear limits:

- orchestration state is fragmented across the host runtime,
- GPU/accelerator locality is poor,
- multi-session scheduling is harder to batch,
- tool control and policy live outside the model substrate,
- the line between "reasoning" and "control" is usually informal.

This repository already points toward a more structured alternative:

- deterministic transformer execution exists,
- explicit VM state exists,
- explicit external I/O seams already exist,
- a proposal lane plus deterministic kernel lane is already the architectural direction.

This roadmap asks whether that architecture can be extended into an agent microkernel without attempting full operating-system replacement.

## One-Year Target

At the end of the roadmap, a credible v1 should be able to do all of the following:

- maintain multiple concurrent agent sessions,
- schedule those sessions with deterministic queues and timers,
- persist and resume session state,
- expose mailboxes, virtual files, and bounded scratch buffers,
- call a probabilistic LLM for reasoning through a typed syscall boundary,
- gate tool use through a deterministic capability table,
- retry, defer, cancel, or rate-limit actions without host intervention,
- step many sessions or candidate plans in parallel on GPU,
- run a demonstrable long-horizon workflow without manual intervention for minutes to hours.

The intended outcome is "something like the control-plane half of `agent-os`" rather than full parity with `agent-os`.

## What this is not

This roadmap does not attempt to:

- replace a conventional host OS or hypervisor,
- move hardware interrupt handling into the transformer,
- reimplement V8, Node.js, or a POSIX shell inside the transformer,
- provide full PTY fidelity,
- provide a real network stack inside the transformer,
- match the entire runtime surface of `agent-os`,
- depend on STARK integration in the initial year.

Proof and attestation should remain a later compatibility target, but they are not required for this roadmap to be valuable.

## Architectural stance

The core design choice is:

- probabilistic reasoning is not the kernel,
- deterministic control is the kernel,
- real-world effects still cross a host boundary.

The proposed system has four layers.

### Layer 1: Host substrate

The host remains responsible for:

- wall-clock time,
- durable persistence,
- actual network and filesystem operations,
- sandboxing and security isolation,
- model loading and device orchestration,
- process lifecycle outside the transformer.

The host should be intentionally thin and event-driven.

### Layer 2: Deterministic transformer microkernel

The kernel owns:

- run queues,
- timer wheel or timer heap,
- message mailboxes,
- capability tables,
- session descriptors,
- scratch buffers and bounded virtual files,
- action validation,
- budget accounting,
- deterministic retries, backoff, and cancellation.

This layer is the policy enforcement point.

### Layer 3: Probabilistic model service

The LLM is exposed as a supervised syscall surface:

- `CALL_MODEL`
- `CALL_MODEL_JSON`
- `RANK_CANDIDATES`
- `SUMMARIZE_STATE`
- `REQUEST_PLAN`

The kernel decides:

- when a model call is allowed,
- which model is used,
- what prompt or context window shape is admissible,
- the token budget,
- the timeout,
- the schema or grammar,
- whether the response is accepted, retried, or discarded.

### Layer 4: Effect bridge

All external effects are represented as explicit typed events:

- timer fired,
- tool result arrived,
- file block read completed,
- network response returned,
- user input appended,
- process exited,
- model call completed.

The kernel never performs raw host callbacks directly.

## Target kernel features

The first year should focus on a compact, closed world of kernel services.

### 1. Session table

Each session should include:

- session id,
- current state / phase,
- pending work item,
- queue membership,
- deadlines and retry counters,
- capability mask,
- mailbox references,
- virtual file references,
- model budget counters.

### 2. Run queues

Support at least:

- ready queue,
- sleeping queue,
- blocked-on-model queue,
- blocked-on-tool queue,
- failed / terminal queue.

Scheduling can start as cooperative and deterministic.

### 3. Timer subsystem

The kernel should support:

- absolute deadlines,
- relative delays,
- retry backoff,
- recurring heartbeats,
- watchdog timeouts.

### 4. Message passing

Support bounded mailboxes with:

- append,
- pop,
- peek,
- wake-on-message,
- overflow policy.

### 5. Virtual file and scratch state

Start with a bounded in-kernel virtual filesystem:

- append-only logs,
- small mutable files,
- named buffers,
- artifact slots,
- transcript windows.

The host can persist these, but the kernel owns the logical semantics.

### 6. Capability table

Every session or task should carry explicit permissions:

- may call model,
- may call tool class X,
- may write file region Y,
- may spawn child session,
- may request elevated budget,
- may access named mailbox Z.

### 7. LLM syscall boundary

The probabilistic model interface should be explicit and typed.

Required metadata:

- model id,
- prompt template id,
- context attachments,
- schema / grammar,
- temperature or decode mode,
- token limits,
- timeout,
- caller session id.

### 8. Checkpoint and resume

The kernel should be able to snapshot and restore:

- session table,
- queues,
- timers,
- virtual file state,
- capability state,
- pending external waits.

### 9. Batched stepping

The kernel runtime should support:

- N independent sessions stepped together,
- N candidate plans for one session stepped together,
- deterministic filtering before committing the next state.

This is one of the main reasons to attempt this architecture.

## The credible v1

The one-year target should be considered successful if it can demonstrate:

1. a deterministic kernel loop running inside the transformer substrate,
2. at least 8 to 64 concurrent sessions stepped in batches,
3. timers, queues, mailboxes, and virtual files working without host-side orchestration logic,
4. probabilistic LLM calls routed through explicit kernel syscalls,
5. resumable long-horizon tasks with no manual intervention,
6. a thin host that mainly injects events and executes approved effects.

## The not-credible v1

The roadmap should be considered off-track if it drifts toward:

- full shell emulation,
- PTY-heavy host behavior becoming the primary control plane,
- Node/V8 reimplementation inside the transformer,
- unconstrained free-form tool use without kernel mediation,
- broad general filesystem semantics,
- large amounts of hidden host logic masquerading as kernel behavior.

## Suggested implementation shape

The safest direction is to add a new additive namespace rather than mutate the current execution core in place.

Suggested modules:

- `src/agent_kernel/mod.rs`
- `src/agent_kernel/state.rs`
- `src/agent_kernel/queue.rs`
- `src/agent_kernel/timer.rs`
- `src/agent_kernel/mailbox.rs`
- `src/agent_kernel/vfs.rs`
- `src/agent_kernel/capability.rs`
- `src/agent_kernel/syscall.rs`
- `src/agent_kernel/effects.rs`
- `src/agent_kernel/runtime.rs`
- `src/agent_kernel/host_bridge.rs`
- `src/agent_kernel/model_call.rs`

If the transformer-native execution path needs a separate ISA or memory map, add it behind an isolated feature or subcommand instead of widening the default TVM path immediately.

## Implementation phases

## Phase 0: architecture freeze

Objective:

Freeze a narrow, credible target before implementation expands.

Deliverables:

- a kernel service inventory,
- a proposed state model,
- a memory map or logical object model,
- a host boundary definition,
- a benchmark plan,
- a rollback plan.

Exit criteria:

- the team can explain the kernel/LLM/host split in one page,
- non-goals are explicit,
- the initial demo scenario is chosen.

## Phase 1: minimal kernel state model

Objective:

Define the deterministic state machine for agent sessions.

Scope:

- session descriptor type,
- queue membership,
- timer records,
- mailbox descriptors,
- virtual file descriptors,
- capability records,
- kernel event enum.

Implementation notes:

- keep counts bounded,
- keep state serializable,
- avoid hidden host-owned fields.

Exit criteria:

- state round-trips cleanly,
- host can inject events without mutating hidden kernel state,
- a single-session kernel loop can advance deterministically.

## Phase 2: queues, timers, and mailboxes

Objective:

Make the kernel useful before adding LLM calls.

Scope:

- ready/sleep/blocked queues,
- deterministic wake and dequeue logic,
- timer heap or timer wheel,
- bounded mailbox append/pop,
- retry and timeout transitions.

Suggested demo:

- multiple toy workers exchanging messages and waking on timers.

Exit criteria:

- scheduler behavior is deterministic,
- timer semantics are well-defined,
- queue operations work under bounded load.

## Phase 3: virtual files and checkpoints

Objective:

Give sessions persistent working memory and resumability.

Scope:

- named virtual files,
- transcript buffers,
- append-only logs,
- snapshots of kernel state,
- resume from serialized snapshot.

Suggested demo:

- an interrupted multi-step task resumes after restore without host-side reconstruction.

Exit criteria:

- save/restore works,
- session-local artifacts persist,
- kernel state is inspectable.

## Phase 4: `CALL_MODEL` syscall

Objective:

Integrate probabilistic reasoning through a typed boundary.

Scope:

- syscall request schema,
- kernel-side budget and timeout validation,
- host-side model invocation adapter,
- structured response handling,
- invalid response and retry policy.

Important rule:

- the model call result is data returned to the kernel,
- the model does not mutate kernel state directly.

Suggested demo:

- a session asks for a plan, receives JSON actions, and the kernel accepts or rejects them.

Exit criteria:

- model calls are explicit in the runtime trace,
- schema violations are recoverable,
- kernel policy, not prompt text, remains the source of control.

## Phase 5: tool syscalls and effect bridge

Objective:

Let sessions perform real work through mediated host tools.

Scope:

- typed tool request events,
- typed tool result events,
- capability-based admission control,
- deterministic pending/complete/error transitions,
- small catalog of tool classes such as file read, file write, HTTP fetch, and shell command wrapper.

Important rule:

- tools return structured results to the kernel,
- tool execution is never a hidden side effect.

Exit criteria:

- the host is mainly an effect executor,
- the kernel decides when and whether tool calls occur,
- sessions can complete nontrivial external workflows.

## Phase 6: batched multi-session stepping

Objective:

Exploit the main architectural advantage of the design.

Scope:

- batch multiple kernel sessions together,
- batch multiple candidate plans for one session,
- measure throughput against a host-centric orchestration baseline,
- keep semantics identical between single-step and batched execution.

Suggested metrics:

- sessions per second,
- average cost per session-step,
- throughput under mixed waiting/runnable workloads,
- cost of blocked-on-model vs blocked-on-tool states.

Exit criteria:

- batch mode is materially useful,
- no semantic divergence from single-session stepping is found,
- scheduling remains explainable.

## Phase 7: long-horizon demo workloads

Objective:

Show the architecture doing work that is actually interesting.

Candidate demos:

- repository maintenance agent with queued subtasks and resumable state,
- literature review agent with bounded notes, retries, and citation queues,
- software migration agent with staged planning, code edits, and validation loops,
- monitoring agent that waits on events, triggers tool runs, and summarizes state periodically.

The demo should visibly require:

- timers,
- mailboxes or queues,
- virtual files,
- model calls,
- tool calls,
- suspension and resume,
- more than one active session.

Exit criteria:

- the system runs for a long horizon without manual steering,
- the host remains thin,
- the value is obvious without invoking proofs.

## Phase 8: hardening and scope decision

Objective:

Decide whether the experiment deserves deeper investment.

Questions to answer:

- does accelerator-native batching produce a meaningful advantage,
- is the host actually thin in practice,
- is the kernel/LLM split stable,
- is the state model small enough to reason about,
- is there a believable path back to attestation later,
- is the approach preferable to a CPU harness for at least one class of workloads.

Possible outcomes:

- continue toward proof integration,
- continue as a practical runtime without proof claims,
- stop and preserve only the useful intermediate artifacts.

## Demo target for the first public milestone

The first milestone worth showing publicly should not be a toy arithmetic loop.

It should be a bounded, real-looking agent workflow such as:

1. ingest a user task,
2. create a session and two helper sessions,
3. schedule model calls and tool calls under explicit budgets,
4. write notes and plans into virtual files,
5. sleep or block while waiting for events,
6. resume automatically,
7. merge results and produce a final artifact.

The point of the demo is to show that:

- the deterministic kernel is directing the probabilistic model,
- long-horizon execution persists without manual intervention,
- orchestration state is not smeared across ad hoc host code.

## Key risks

1. Too much hidden host logic may creep back in.
2. Latency may dominate unless batching is real and frequent.
3. Kernel state may become too wide or too dynamic to keep understandable.
4. Model-call schemas may be too brittle for practical use.
5. Virtual filesystem semantics may grow into a fake POSIX layer and consume the roadmap.
6. The design may not outperform a well-built CPU harness for practical workloads.

## Kill criteria

Stop or narrow the effort if any of the following becomes true:

1. The host still owns most control logic after Phase 5.
2. Batch stepping does not provide a material throughput or cost advantage.
3. The kernel state model becomes too complex to explain cleanly.
4. The system requires full shell or V8 compatibility to be useful.
5. Long-horizon tasks still need frequent manual intervention.
6. The architecture cannot support a clear typed boundary between kernel control and LLM proposals.

## Relationship to proof work

This roadmap intentionally sets proofs aside.

However, it should avoid choices that would block later attestation work. In particular:

- keep kernel state explicit,
- keep events typed,
- keep external effects logged rather than implicit,
- keep model and tool calls as explicit syscalls,
- keep snapshots serializable and hashable.

That keeps the door open for future receipts or proofs without forcing them into year one.

## Bottom line

The credible one-year target is not "an operating system in the transformer."

It is:

- a deterministic transformer-native agent microkernel,
- supervising probabilistic LLM reasoning,
- mediating tools and external effects through a thin host,
- supporting resumable long-horizon tasks,
- and exploiting accelerator-native batching where a CPU harness usually cannot.

That is ambitious, but it is materially more credible than full in-transformer `agent-os` parity, and it is large enough to be genuinely interesting if it works.
