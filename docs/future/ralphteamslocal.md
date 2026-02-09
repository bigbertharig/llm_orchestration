# Replicating Agent Teams and Ralph Loops on a Local LLM Cluster

An analysis of how to implement both paradigms using local models on consumer GPU hardware.

---

## The Core Challenge

Claude's Agent Teams and Ralph Loops both rely on capabilities that frontier models excel at:
- Long context windows (200K-1M tokens)
- Strong instruction following
- Reliable tool use / function calling
- Self-awareness of task completion

Local models (7B-14B parameters on 6GB GPUs) are weaker in all these areas. This document explores adaptations that work within those constraints.

---

## Part 1: Local Ralph Loops

### Why Ralph Loops Are More Feasible Locally

Ralph loops are simpler to replicate because:
1. **Single agent** - no coordination overhead
2. **Disk-based state** - context stored in files/git, not conversation history
3. **External verification** - completion determined by tests/linters, not model judgment
4. **Fresh context each iteration** - avoids accumulating context rot

### Architecture for Local Ralph

```
┌─────────────────────────────────────────────────────────┐
│                    RALPH CONTROLLER                      │
│                    (Python script)                       │
├─────────────────────────────────────────────────────────┤
│  1. Load PROMPT.md (task definition)                    │
│  2. Gather context: git diff, changed files, test output│
│  3. Build prompt with current state                     │
│  4. Send to local LLM                                   │
│  5. Parse response, apply file changes                  │
│  6. Run verification (tests, linter, type check)        │
│  7. Check exit conditions                               │
│  8. Loop or exit                                        │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│              LOCAL LLM (Ollama/vLLM)                    │
│              Qwen 7B or similar                         │
│              Single GPU                                 │
└─────────────────────────────────────────────────────────┘
```

### Key Adaptations for Local Models

#### 1. Aggressive Context Pruning
Local models have 4K-32K context windows vs Claude's 200K+. Each iteration must:
- Include only the most recent test failures (not full history)
- Show only changed files, not entire codebase
- Summarize previous attempts rather than including full transcripts
- Use a sliding window of the last N iterations' summaries

#### 2. Simpler Task Granularity
Break tasks into smaller, atomic units:
- Instead of "migrate entire test suite from Jest to Vitest"
- Use "migrate tests in src/utils/*.test.js from Jest to Vitest"

The controller can loop through file groups sequentially.

#### 3. Structured Output Enforcement
Local models are less reliable at following complex output formats. Use:
- JSON mode where available (Ollama supports this)
- Simple, rigid templates: `FILE: path\n---\nCONTENT\n---`
- Regex-based parsing with fallback prompting on parse failure
- Grammar-constrained generation (llama.cpp GBNF)

#### 4. Explicit Completion Signals
Don't rely on the model to self-assess completion. Instead:
- Run tests after each iteration: exit 0 = done
- Check for specific file markers or patterns
- Count consecutive "no changes needed" responses
- Set hard iteration limits (e.g., max 20 iterations)

#### 5. Error Recovery
Local models get stuck more often. Build in:
- Detection of repeated identical outputs (loop stuck)
- Backoff and retry with rephrased prompt
- Git reset to last known good state after N failures
- Escalation path to larger model or human

### Local Ralph Implementation Sketch

```python
# Pseudocode for local Ralph loop

MAX_ITERATIONS = 30
STUCK_THRESHOLD = 3

def run_ralph(prompt_file, project_dir):
    iteration = 0
    stuck_count = 0
    last_output_hash = None

    while iteration < MAX_ITERATIONS:
        # Build context (keep it small)
        context = gather_context(project_dir, max_files=5, max_lines=500)
        test_output = run_tests(project_dir)

        # Check if done
        if test_output.exit_code == 0 and all_types_resolve(project_dir):
            return "SUCCESS"

        # Build prompt
        prompt = f"""
Task: {read_file(prompt_file)}

Current test failures:
{truncate(test_output.stderr, 1000)}

Recently changed files:
{context}

Fix the issues. Output your changes as:
FILE: <path>
---
<content>
---
"""

        # Call local LLM
        response = ollama_generate("qwen2.5-coder:7b", prompt)

        # Detect stuck loop
        output_hash = hash(response)
        if output_hash == last_output_hash:
            stuck_count += 1
            if stuck_count >= STUCK_THRESHOLD:
                response = ollama_generate("qwen2.5-coder:7b",
                    prompt + "\n\nPrevious approach failed. Try a different strategy.")
                stuck_count = 0
        else:
            stuck_count = 0
        last_output_hash = output_hash

        # Apply changes
        apply_file_changes(response, project_dir)
        git_commit(project_dir, f"ralph iteration {iteration}")

        iteration += 1

    return "MAX_ITERATIONS_REACHED"
```

### Hardware Considerations

| Scenario | GPU Allocation | Notes |
|----------|----------------|-------|
| Single Ralph loop | 1x GPU with 7B model | Most efficient |
| Parallel Ralph loops | 1 GPU per loop | Different projects or file groups |
| Ralph with escalation | 1x GPU for 7B, 2x GPU for 14B | Escalate on repeated failures |

---

## Part 2: Local Agent Teams

### Why Agent Teams Are Harder Locally

Agent Teams face significant challenges on local hardware:
1. **Multiple concurrent contexts** - each agent needs its own model instance
2. **Inter-agent messaging** - requires parsing and routing infrastructure
3. **Coordination overhead** - lead agent must understand team state
4. **Context limits** - harder to maintain coherent multi-agent conversations

### Feasible Local Team Architectures

#### Option A: Time-Sliced Single Model

One model instance, agents take turns:

```
┌─────────────────────────────────────────────────────────┐
│                   TEAM COORDINATOR                       │
│                   (Python process)                       │
├─────────────────────────────────────────────────────────┤
│  Shared Task List (filesystem)                          │
│  Agent Mailboxes (filesystem)                           │
│  Turn Scheduler                                         │
└─────────────────────────────────────────────────────────┘
           │
           ▼ (one at a time)
┌─────────────────────────────────────────────────────────┐
│              SINGLE LLM INSTANCE                        │
│              Context switches per agent                 │
└─────────────────────────────────────────────────────────┘
```

**Pros:** Low resource usage, simple
**Cons:** No parallelism, slow, context switching overhead

#### Option B: Multi-GPU Parallel Agents

Each agent runs on its own GPU:

```
┌─────────────────────────────────────────────────────────┐
│                   TEAM COORDINATOR                       │
├─────────────────────────────────────────────────────────┤
│  Shared Task List    │  Message Router                  │
│  (Redis/filesystem)  │  (ZeroMQ/HTTP)                   │
└─────────────────────────────────────────────────────────┘
     │           │           │           │
     ▼           ▼           ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ Agent 0 │ │ Agent 1 │ │ Agent 2 │ │ Agent 3 │
│ (Lead)  │ │ Worker  │ │ Worker  │ │ Worker  │
│ GPU 0+3 │ │ GPU 1   │ │ GPU 2   │ │ GPU 4   │
│ 14B     │ │ 7B      │ │ 7B      │ │ 7B      │
└─────────┘ └─────────┘ └─────────┘ └─────────┘
```

**Pros:** True parallelism, matches Claude's architecture
**Cons:** Needs 4+ GPUs, complex coordination, VRAM hungry

#### Option C: Hybrid - Coordinator + Worker Pool

Smarter coordinator, simpler workers:

```
┌─────────────────────────────────────────────────────────┐
│              LEAD AGENT (14B on 2 GPUs)                 │
│  - Breaks down tasks                                    │
│  - Assigns to workers                                   │
│  - Synthesizes results                                  │
│  - Makes judgment calls                                 │
└─────────────────────────────────────────────────────────┘
           │
           ▼ (task queue)
┌─────────────────────────────────────────────────────────┐
│                    WORKER POOL                          │
│  3x 7B models on separate GPUs                          │
│  - Execute assigned tasks                               │
│  - Report results back                                  │
│  - No inter-worker communication                        │
└─────────────────────────────────────────────────────────┘
```

**Pros:** Plays to model strengths, reduces coordination complexity
**Cons:** Workers can't collaborate directly, lead becomes bottleneck

### Key Adaptations for Local Teams

#### 1. Filesystem-Based Coordination
Avoid complex message passing. Use files:
```
shared/
  teams/
    {team-id}/
      config.json          # Team structure
      tasks/
        pending/           # Unclaimed tasks
        in_progress/       # Claimed tasks
        complete/          # Done tasks
      mailbox/
        agent-0/           # Messages for agent 0
        agent-1/           # Messages for agent 1
```

File locking prevents race conditions on task claiming.

#### 2. Simplified Agent Roles
Local models struggle with nuanced role-playing. Use explicit, distinct roles:
- **Lead:** Only coordinates, never implements
- **Coder:** Only writes code, doesn't review
- **Reviewer:** Only reviews, doesn't write
- **Tester:** Only runs and analyzes tests

Each role has a focused system prompt and limited tool access.

#### 3. Structured Message Format
Inter-agent messages must be parseable:
```json
{
  "from": "agent-1",
  "to": "agent-0",
  "type": "task_complete",
  "task_id": "task-42",
  "summary": "Implemented login function",
  "files_changed": ["src/auth/login.py"],
  "needs_review": true
}
```

#### 4. Synchronization Points
Don't let agents run indefinitely. Use checkpoints:
- After each task completion, agents report to lead
- Lead reviews progress every N minutes
- Hard timeouts force status updates
- Circuit breaker if an agent produces no output for M minutes

#### 5. Conflict Prevention
Assign file ownership explicitly:
```json
{
  "task_id": "task-42",
  "assigned_to": "agent-1",
  "owns_files": ["src/auth/*"],
  "read_only": ["src/utils/*"]
}
```

Coordinator rejects tasks that would create conflicts.

### Local Team Implementation Sketch

```python
# Pseudocode for local agent team coordinator

class LocalAgentTeam:
    def __init__(self, team_dir, gpu_allocation):
        self.team_dir = team_dir
        self.agents = {}
        self.task_queue = FilesystemTaskQueue(team_dir / "tasks")
        self.mailbox = FilesystemMailbox(team_dir / "mailbox")

    def spawn_lead(self, gpu_ids):
        """Lead uses larger model on multiple GPUs"""
        self.agents["lead"] = Agent(
            name="lead",
            model="qwen2.5:14b",
            gpu_ids=gpu_ids,
            role="coordinator",
            system_prompt=LEAD_PROMPT
        )

    def spawn_worker(self, name, gpu_id, role):
        """Workers use smaller model on single GPU"""
        self.agents[name] = Agent(
            name=name,
            model="qwen2.5-coder:7b",
            gpu_ids=[gpu_id],
            role=role,
            system_prompt=WORKER_PROMPTS[role]
        )

    def run(self, initial_task):
        # Lead breaks down initial task
        subtasks = self.agents["lead"].plan(initial_task)
        for task in subtasks:
            self.task_queue.add(task)

        # Main loop
        while not self.task_queue.all_complete():
            # Each worker claims and works on tasks
            for name, agent in self.agents.items():
                if name == "lead":
                    continue

                # Check for messages
                messages = self.mailbox.get(name)
                for msg in messages:
                    agent.receive(msg)

                # Claim task if idle
                if agent.is_idle():
                    task = self.task_queue.claim(name)
                    if task:
                        agent.start_task(task)

                # Do one step of work
                result = agent.step()

                # Handle completion
                if result.task_complete:
                    self.task_queue.complete(result.task_id)
                    self.mailbox.send("lead", {
                        "from": name,
                        "type": "task_complete",
                        "task_id": result.task_id,
                        "summary": result.summary
                    })

            # Lead synthesizes and maybe spawns new tasks
            self.agents["lead"].coordinate(
                self.mailbox.get("lead"),
                self.task_queue.status()
            )
```

### Resource Requirements

| Configuration | GPUs | Models | Parallelism |
|---------------|------|--------|-------------|
| Minimal team | 2 | 1x14B lead, 1x7B worker | Low |
| Balanced | 4 | 1x14B lead, 3x7B workers | Medium |
| Full cluster | 5 | 1x14B lead (2 GPU), 3x7B workers | High |

---

## Part 3: Hybrid Approach - Ralph Teams

Combine both paradigms: teams of Ralph loops.

```
┌─────────────────────────────────────────────────────────┐
│                   META-COORDINATOR                       │
│          Assigns file groups to Ralph loops             │
└─────────────────────────────────────────────────────────┘
           │
     ┌─────┴─────┬───────────┐
     ▼           ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ Ralph 1 │ │ Ralph 2 │ │ Ralph 3 │
│ src/api │ │ src/ui  │ │ tests/  │
│ GPU 1   │ │ GPU 2   │ │ GPU 4   │
└─────────┘ └─────────┘ └─────────┘
```

Each Ralph loop:
- Owns a specific directory/file set
- Runs independently until tests pass
- Reports completion to meta-coordinator
- Cannot touch files owned by other loops

This gets parallelism without inter-agent communication complexity.

---

## Recommendations by Use Case

| Task Type | Recommended Approach | Why |
|-----------|---------------------|-----|
| Large refactor | Parallel Ralph loops | Divide by file group, run in parallel |
| Code review | Time-sliced team | Different reviewer perspectives, sequential |
| New feature | Hybrid coordinator + Ralph | Lead plans, workers Ralph their modules |
| Debugging | Single Ralph with escalation | Needs coherent hypothesis tracking |
| Test coverage | Parallel Ralph loops | Each loop covers different test files |

---

## Limitations vs Claude Implementation

| Capability | Claude | Local (5x 1060 6GB) |
|------------|--------|---------------------|
| Context window | 200K-1M tokens | 4K-32K tokens |
| Instruction following | Excellent | Good (7B), Better (14B) |
| Tool use reliability | High | Medium |
| Parallel agents | 10+ | 3-4 (VRAM limited) |
| Self-assessment | Reliable | Unreliable |
| Inter-agent reasoning | Sophisticated | Basic |

### Mitigations

1. **Short context** → Disk-based state, aggressive pruning, iteration summaries
2. **Weak instruction following** → Structured output, grammar constraints, retries
3. **Limited agents** → Prioritize independent tasks, reduce coordination
4. **Poor self-assessment** → External verification only (tests, linters)
5. **Basic reasoning** → Simpler task decomposition, more explicit prompts

---

## Next Steps for Implementation

1. **Start with local Ralph** - Simplest, highest ROI
2. **Add escalation tier** - 7B → 14B → Claude (via gateway)
3. **Implement parallel Ralph** - One loop per GPU, file ownership
4. **Build team coordinator** - If multi-perspective work is needed
5. **Hybrid approach** - Lead plans, Ralph loops execute

---

*Analysis based on Claude Agent Teams (Opus 4.6, Feb 2026) and Ralph Loop methodology (late 2025), adapted for consumer GPU hardware with 7B-14B parameter models.*
