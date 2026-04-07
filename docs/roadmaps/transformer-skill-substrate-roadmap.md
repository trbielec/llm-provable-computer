# Transformer Skill Substrate Roadmap

## Title

Future-possible roadmap for a transformer-native deterministic skill toolkit

## Status

- Date: April 6, 2026
- Horizon: 6 to 12 months of focused exploratory engineering
- Status: speculative and additive
- Relationship to current work: complements `docs/roadmaps/current-roadmap.md`
- Relationship to adjacent roadmap: narrower and earlier than `docs/roadmaps/transformer-agent-microkernel-roadmap.md`

## Product statement

Build a deterministic skill substrate inside the transformer VM so that a probabilistic LLM can route tasks to compiled, GPU-local, exact programs instead of outsourcing every capability to a CPU harness or ad hoc script tool.

The target is not:

- "compile arbitrary Python into the transformer,"
- "move all tools into attention,"
- or "port the whole PSN runtime into the proof-critical core."

The target is narrower:

- define a typed deterministic skill profile,
- lower skills into a compact IR and then into TVM / transformer-executable programs,
- let a probabilistic or deterministic router select which skill or skill composition to run,
- execute the selected program deterministically inside the transformer-shaped runtime,
- return structured outputs to the agent loop.

## Why this exists

The current tool-using LLM pattern is usually:

- route from natural language to a tool call,
- leave the GPU,
- run Python, shell, or a service,
- bring results back into the model loop,
- repeat many times.

That works, but it has costs:

- orchestration bounces between CPU and GPU,
- deterministic subroutines are not accelerator-local,
- tool semantics are often informal,
- routing and execution are mixed together,
- many small "skills" are implemented as arbitrary scripts rather than typed, bounded programs.

The `RLM` / PSN work suggests a more structured direction:

- skills are explicit programs,
- skills carry interfaces and conditions,
- skills form a graph,
- routing can use backward-chaining and value-based selection,
- missing capabilities can still be discovered by an outer probabilistic layer.

The key adaptation for this repo is:

- keep the router and synthesis logic outside the deterministic transformer core,
- move a bounded, typed, high-value subset of deterministic skills inside it.

## Core hypothesis

A probabilistic LLM does not need to perform every step of a complex task itself if it has access to a large library of exact, accelerator-local programs that it can invoke compositionally.

In this framing:

- the LLM is the router, decomposer, and semantic planner,
- the transformer VM is the deterministic skill substrate,
- the host remains a thin effect executor for anything not representable inside the substrate.

## Relationship to PSN

This roadmap is strongly inspired by the Programmatic Skill Network shape in `RLM`, but it intentionally narrows the scope.

Good PSN ideas to import:

- skills as explicit programs,
- explicit input/output contracts,
- postcondition-aware matching,
- invocation graph structure,
- maturity or reliability scoring,
- hierarchical composition,
- reusable skill library as the main capability surface.

PSN features that should remain outside the deterministic core:

- arbitrary Python execution,
- dynamic code generation into raw host scripts,
- PATCH-driven autonomous code rewriting,
- LLM-native semantic fallback inside the kernel,
- embedding-heavy candidate discovery in the inner loop,
- unrestricted filesystem-based persistence as the skill substrate.

## One-Year Target

At the end of the roadmap, a credible v1 should be able to do all of the following:

- maintain a catalog of 100 to 1000 deterministic skills in a typed registry,
- compile a bounded subset of those skills into TVM / transformer-executable programs,
- route tasks to one skill or a short skill composition plan,
- execute the plan deterministically on-GPU,
- return structured outputs and intermediate artifacts,
- score skills by reliability and cost,
- batch candidate skill runs or parameterized skill invocations,
- outperform a host-script tool loop for at least one class of workloads.

The key value is not "all tools live inside the transformer."

The key value is:

- the highest-frequency deterministic cognitive subroutines become GPU-local.

## What this is not

This roadmap does not attempt to:

- compile arbitrary Python or arbitrary shell commands,
- compile unbounded recursion or unbounded dynamic data structures,
- support unrestricted side effects inside the transformer,
- replace general host tools,
- replace the later microkernel roadmap,
- absorb semantic routing and skill synthesis into the deterministic core,
- broaden proof claims in year one.

## Architectural stance

The proposed system has five layers.

### Layer 1: Skill authoring layer

Skills begin as typed specifications, not opaque scripts.

Each skill should have:

- name,
- description,
- typed inputs,
- typed outputs,
- preconditions,
- postconditions,
- cost bound,
- effect class,
- implementation profile.

### Layer 2: Skill IR

A new intermediate representation should capture a bounded deterministic skill program.

The IR should support:

- typed scalar values,
- bounded lists or arrays,
- records,
- conditionals,
- bounded loops,
- calls to approved intrinsics,
- explicit success/failure outputs,
- optional structured trace data.

### Layer 3: Deterministic transformer skill runtime

This runtime executes compiled skills via:

- native interpreter,
- current transformer-shaped runtime,
- optional batch execution backends.

This is the execution substrate.

### Layer 4: Router / planner

A probabilistic or deterministic router selects:

- one skill,
- one parameterization,
- or a short composition graph.

This layer may use:

- natural-language matching,
- PSN-style backward-chaining,
- maturity or value scores,
- cost heuristics,
- local or remote LLMs.

This layer should not define the deterministic skill semantics.

### Layer 5: Effect bridge

If a skill needs external data or host-side action, it should do so through explicit effect classes:

- pure deterministic,
- read-only external,
- write external,
- model-call-assisted,
- host-only fallback.

## The right split

The clean split is:

- deterministic substrate for exact skills,
- probabilistic router for semantic selection,
- thin host for real-world effects.

That means a task like "analyze this document and extract stakeholder roles" might become:

1. router chooses document-analysis skill chain,
2. substrate executes chunking, extraction, counting, filtering, and aggregation deterministically,
3. router optionally asks LLM for semantic synthesis over the deterministic intermediate outputs.

## The wrong split

The architecture is likely to fail if it becomes:

- LLM routes to arbitrary Python code,
- Python code is treated as the skill substrate,
- the host quietly owns most execution semantics,
- transformer execution is only a marketing layer rather than the real deterministic capability surface.

## Skill profile design

The first year needs a strict skill profile.

### Profile A: transformer-native pure skills

Allowed:

- arithmetic,
- comparisons,
- filtering,
- bounded reduction,
- extraction,
- token or byte transforms,
- fixed-schema parsing,
- bounded graph or tree traversal,
- scoring,
- aggregation,
- exact selection or ranking.

Disallowed:

- arbitrary file I/O,
- arbitrary dynamic imports,
- unrestricted recursion,
- unbounded memory allocation,
- host callbacks during execution.

### Profile B: external-read skills

Adds:

- read-only effect inputs,
- explicit external data handles,
- bounded fetched context,
- deterministic execution after input materialization.

### Profile C: composed skills with host boundary

Adds:

- skill plans mixing compiled skills with host-only fallback nodes,
- explicit router decisions around those boundaries,
- deterministic subgraphs inside larger mixed workflows.

The first public version should focus on Profile A.

## Suggested data model

### `SkillSpec`

A high-level authored skill definition:

- `id`
- `name`
- `description`
- `input_schema`
- `output_schema`
- `preconditions`
- `postconditions`
- `effect_class`
- `cost_class`
- `implementation`

### `CompiledSkill`

A lowered, executable artifact:

- `skill_id`
- `profile`
- `program_hash`
- `entry_point`
- `memory_layout`
- `resource_bounds`
- `input_adapter`
- `output_adapter`

### `SkillPlan`

A router-selected plan:

- ordered or DAG-shaped list of steps,
- concrete parameter bindings,
- expected outputs,
- cost estimate,
- fallback policy.

### `SkillExecutionRecord`

Execution metadata:

- skill id,
- inputs,
- outputs,
- success/failure,
- runtime cost,
- selected fallback if any,
- optional receipt handle later.

## Recommended modules

Suggested additive module structure:

- `src/skills/mod.rs`
- `src/skills/spec.rs`
- `src/skills/schema.rs`
- `src/skills/registry.rs`
- `src/skills/ir.rs`
- `src/skills/lowering.rs`
- `src/skills/runtime.rs`
- `src/skills/adapter.rs`
- `src/skills/profile.rs`
- `src/skills/router_types.rs`
- `src/skills/catalog.rs`

If a separate crate is cleaner:

- `crates/tvm-skills-core`
- `crates/tvm-skills-router`

## Candidate initial domains

The first compiled skill catalog should focus on domains with high deterministic density.

Good initial domains:

- text slicing and chunking,
- structured extraction from bounded formats,
- counting and aggregation,
- regex-like or token-pattern filtering,
- table transforms,
- simple statistics,
- ranking and scoring,
- citation extraction,
- transcript parsing,
- bounded planning over explicit graphs.

Poor initial domains:

- unconstrained code generation,
- broad semantic writing,
- open-ended research synthesis,
- shell-heavy workflows,
- network-heavy workflows,
- very large-context reasoning that is mostly generative.

## Example migration path from PSN

PSN skills should be classified into three buckets.

### Bucket 1: compile now

Examples:

- count,
- filter,
- extract pattern,
- aggregate,
- compute statistics,
- transform bounded structures.

These become `SkillSpec -> SkillIR -> TVM`.

### Bucket 2: router-only metadata

Examples:

- composite skills that mostly express decomposition,
- higher-level analysis templates,
- capability descriptions.

These remain router-visible but are not yet compiled.

### Bucket 3: host-only fallback

Examples:

- arbitrary Python scripts,
- filesystem-heavy or library-heavy utilities,
- skills relying on unrestricted code execution.

These stay outside until a narrower deterministic profile exists.

## Implementation phases

## Phase 0: freeze the skill profile

Objective:

Define the exact kind of skills that are allowed into the deterministic substrate.

Deliverables:

- skill profile document,
- initial `SkillSpec` schema,
- effect-class taxonomy,
- candidate skill shortlist,
- non-goals and kill criteria.

Exit criteria:

- the team can classify candidate skills consistently,
- the first 25 to 50 skills are identified,
- arbitrary Python is explicitly out of scope.

## Phase 1: typed skill registry

Objective:

Create a typed catalog before compiling anything.

Scope:

- `SkillSpec`,
- schema validation,
- registry storage,
- metadata for costs and effects,
- loader for a curated initial catalog.

Suggested input sources:

- manually authored specs,
- selected PSN skills rewritten as specs,
- generated inventory from `psn_skills_db` with human triage.

Exit criteria:

- a registry of curated skills exists,
- every included skill has typed inputs and outputs,
- compile eligibility can be decided mechanically.

## Phase 2: Skill IR and lowering

Objective:

Create an IR narrow enough to compile reliably.

Scope:

- scalar and record operations,
- bounded collections,
- intrinsics,
- conditionals,
- bounded loops,
- deterministic failure paths,
- adapters to existing TVM concepts.

Important rule:

- the IR should describe skill semantics, not natural-language routing semantics.

Exit criteria:

- several hand-authored skills lower cleanly into IR,
- the IR is auditable and small,
- resource bounds can be computed up front.

## Phase 3: compiled execution path

Objective:

Execute compiled skills through the existing runtimes.

Scope:

- native interpreter execution,
- transformer-shaped execution,
- adapters from typed input to VM memory,
- adapters from final memory to typed output.

Suggested demo:

- run 10 to 20 compiled document or numeric skills end-to-end with exact output matching.

Exit criteria:

- compiled skills run deterministically,
- inputs and outputs are reproducible,
- transformer and native paths agree on the supported profile.

## Phase 4: skill composition

Objective:

Allow short deterministic compositions without involving the LLM at every step.

Scope:

- linear or DAG-shaped skill plans,
- typed intermediate artifacts,
- local failure handling,
- bounded plan depth,
- reusable subplans.

Important rule:

- composition should remain explicit and typed,
- hidden prompt glue should not become the composition language.

Exit criteria:

- multi-step skill pipelines execute without host-script orchestration,
- intermediate artifacts remain structured,
- plan semantics are deterministic.

## Phase 5: router integration

Objective:

Let a probabilistic or deterministic router target the compiled substrate.

Scope:

- skill retrieval by description,
- skill ranking by cost and maturity,
- backward-chaining over postconditions,
- parameter binding,
- optional LLM planner for goal decomposition,
- host fallback when no compiled skill applies.

Suggested design rule:

- the router outputs a `SkillPlan`,
- the substrate executes it,
- the router does not directly execute arbitrary code.

Exit criteria:

- natural-language tasks can route to compiled skills,
- simple PSN-style matching works,
- the router/substrate boundary is explicit and stable.

## Phase 6: batched candidate evaluation

Objective:

Exploit the main performance advantage of the architecture.

Scope:

- batch multiple skill invocations,
- batch multiple candidate plans,
- deterministically compare outputs,
- measure throughput vs CPU-host tool loops.

Suggested benchmarks:

- extraction workload,
- transcript analysis workload,
- structured statistics workload,
- candidate-plan scoring workload.

Exit criteria:

- batch mode is materially useful,
- at least one workload class shows a real advantage,
- semantics stay identical to single-run execution.

## Phase 7: PSN-inspired reliability and maturity

Objective:

Bring a subset of PSN network semantics into routing without absorbing the full PSN stack.

Scope:

- skill reliability scores,
- maturity-aware ranking,
- postcondition coverage tracking,
- invocation graph stats,
- optional deterministic deprecation or promotion rules.

Important rule:

- do not add self-modifying compiled skills in this phase,
- keep learning and synthesis outside the compiled core.

Exit criteria:

- the router prefers reliable compiled skills,
- skill graph metadata improves routing quality,
- the system remains understandable.

## Phase 8: mixed workflows and host fallback

Objective:

Make the system useful on real tasks without pretending everything is compileable.

Scope:

- mixed plans with compiled and host-only nodes,
- explicit fallback markers,
- partial deterministic execution within a larger workflow,
- structured outputs suitable for later summarization by the LLM.

Suggested demo:

- a document-analysis task where the bulk of transformation work is compiled, and only final semantic synthesis is left to the LLM.

Exit criteria:

- compiled skills carry a visible share of the actual work,
- fallback remains explicit,
- the architecture is still cleaner than a pure host-script tool loop.

## Demo target

The first public demo worth showing should look like:

1. the user gives a task in natural language,
2. the router selects a short skill plan from a catalog,
3. most deterministic transformation happens in compiled on-GPU skills,
4. the LLM is invoked only where semantic judgment is actually needed,
5. the final output cites which compiled skills were used.

Good example tasks:

- transcript analysis,
- structured document extraction,
- research-note aggregation,
- bounded compliance or policy checking,
- deterministic preprocessing before final synthesis.

## Metrics that matter

This roadmap should be judged by:

- percentage of workflow steps handled by compiled deterministic skills,
- GPU-local execution ratio,
- router accuracy,
- skill reuse rate,
- latency and throughput on batched workloads,
- reduction in host orchestration complexity,
- determinism and repeatability.

## Risks

1. The skill profile may be too narrow to be useful.
2. The compilation pipeline may be too expensive to maintain.
3. Natural-language routing may remain the real bottleneck.
4. Too many PSN skills may depend on arbitrary Python semantics.
5. Host fallback may dominate, reducing the value of the compiled substrate.
6. The project may drift into building a second general-purpose programming language without enough payoff.

## Kill criteria

Stop or narrow the effort if any of the following becomes true:

1. Fewer than a meaningful core of high-value skills can be compiled cleanly.
2. Host-only fallback still performs most useful work after Phase 8.
3. Routing quality is too poor to target the substrate reliably.
4. The compiled profile becomes too broad to explain or verify.
5. There is no measurable advantage over a plain CPU tool harness for the selected workloads.

## Relationship to proof work

This roadmap sets proofs aside, but it should remain compatible with later attestation work.

That means:

- typed inputs and outputs,
- explicit skill identity,
- bounded execution,
- explicit effect classes,
- structured execution records,
- deterministic compiled semantics.

If those are preserved, the skill substrate can later inherit receipts or proofs without redesigning the whole system.

## Relationship to the microkernel roadmap

The skill-substrate roadmap is likely earlier and easier than the microkernel roadmap.

If both are pursued, the likely dependency direction is:

- first build compiled deterministic skills,
- then let the later transformer-native agent microkernel treat those skills as one class of supervised syscall.

In that combined architecture:

- the microkernel handles scheduling and long-horizon control,
- the skill substrate provides exact, GPU-local deterministic capability,
- the LLM handles semantic routing and open-ended reasoning.

## Bottom line

The most credible near-term path is not "put all of PSN into the transformer."

It is:

- extract the deterministic core of a PSN-style toolkit,
- represent it as typed skills,
- compile that subset into the transformer VM,
- let a probabilistic router target that exact capability surface,
- and keep synthesis, repair, and semantic fallback outside the deterministic core.

That is ambitious but materially more plausible than compiling arbitrary tool ecosystems, and it creates a concrete bridge between PSN-style skill graphs and transformer-native deterministic execution.
