# Upgrading LLM Orchestration for Ralph Loops and Agent Teams

This document describes how to add Ralph Loop and Agent Team capabilities to the existing llm_orchestration system, building on the current brain/worker architecture.

---

## Current System Strengths

The existing system already has solid foundations that align with both paradigms:

| Existing Feature | Ralph Relevance | Agent Teams Relevance |
|-----------------|-----------------|----------------------|
| Brain/Worker hierarchy | Escalation path for stuck loops | Lead/worker coordination |
| Filesystem task queue | Disk-based state between iterations | Shared task list |
| Dependency tracking (`depends_on`) | Task sequencing | Inter-task coordination |
| Training data logging | Loop improvement over time | Agent behavior analysis |
| Permission system | Sandbox autonomous loops | Agent access control |
| GPU resource manager design | Exclusive batch mode | Parallel agent allocation |

---

## Part 1: Adding Ralph Loop Capability

### 1.1 New Component: Ralph Controller

Create `scripts/ralph.py` - a loop controller that wraps the existing system:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RALPH CONTROLLER                             │
│                        scripts/ralph.py                             │
├─────────────────────────────────────────────────────────────────────┤
│  1. Load PROMPT.md (task definition + completion criteria)          │
│  2. Gather context: git status, test output, changed files          │
│  3. Submit task to existing brain/worker system                     │
│  4. Wait for completion                                             │
│  5. Run verification (tests, linter, type check)                    │
│  6. Check exit conditions:                                          │
│     - All tests pass? EXIT                                          │
│     - Max iterations reached? EXIT                                  │
│     - Stuck detection (same output 3x)? ESCALATE or EXIT            │
│  7. If not done: inject feedback, loop to step 2                    │
└─────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│              EXISTING BRAIN/WORKER SYSTEM                           │
│              (unchanged - ralph.py is a client)                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Ralph Project Structure

```
shared/ralph/
├── {project-name}/
│   ├── PROMPT.md           # Task definition (required)
│   ├── COMPLETION.md       # Exit criteria (required)
│   ├── .ralphrc            # Configuration (optional)
│   ├── iterations/
│   │   ├── 001/
│   │   │   ├── context.json    # Git state, file hashes
│   │   │   ├── task.json       # Submitted task
│   │   │   ├── result.json     # Brain/worker response
│   │   │   └── verification.json # Test results
│   │   ├── 002/
│   │   └── ...
│   └── state.json          # Current iteration, stuck count, etc.
```

### 1.3 PROMPT.md Format

```markdown
# Ralph Task: {Name}

## Objective
{Clear description of what needs to be accomplished}

## Working Directory
{Path to the codebase being modified}

## Files to Modify
- src/auth/*.py
- tests/test_auth.py

## Constraints
- Do not modify src/core/*.py
- Keep all existing tests passing
- Follow existing code style

## Current Status
{This section is auto-updated each iteration with:}
- Files changed in last iteration
- Current test failures
- Linter warnings
```

### 1.4 COMPLETION.md Format

```markdown
# Completion Criteria

## Required (all must pass)
- [ ] `pytest tests/` exits with code 0
- [ ] `mypy src/` reports no errors
- [ ] `ruff check src/` reports no errors

## Optional (nice to have)
- [ ] Test coverage > 80%
- [ ] No TODO comments in changed files

## Stuck Detection
- Max iterations: 20
- Same output threshold: 3
- On stuck: escalate_to_brain
```

### 1.5 Ralph Configuration (.ralphrc)

```json
{
  "max_iterations": 20,
  "stuck_threshold": 3,
  "verification_commands": [
    {"name": "tests", "command": "pytest tests/", "required": true},
    {"name": "types", "command": "mypy src/", "required": true},
    {"name": "lint", "command": "ruff check src/", "required": false}
  ],
  "on_stuck": "escalate",
  "on_max_iterations": "stop",
  "context_strategy": "git_diff",
  "task_target": "worker",
  "escalation_target": "brain",
  "git_commit_each_iteration": true
}
```

### 1.6 Integration with Existing System

Ralph uses the existing task queue - no changes needed to brain.py or worker.py:

```python
# ralph.py submits tasks like any other client
def submit_iteration(self, prompt: str, context: str):
    task = {
        "task_id": str(uuid.uuid4()),
        "type": "generate",  # or "shell" for direct commands
        "prompt": prompt,
        "context": context,
        "priority": 4,
        "created_by": f"ralph-{self.project_name}",
        "ralph_iteration": self.current_iteration
    }

    task_file = self.queue_path / f"{task['task_id']}.json"
    with open(task_file, 'w') as f:
        json.dump(task, f, indent=2)

    return task["task_id"]
```

### 1.7 Escalation Path

When Ralph gets stuck, it can escalate to the brain for help:

```
Worker (7B) attempts fix
    ↓ stuck 3x
Brain (14B) analyzes and suggests approach
    ↓ still stuck
Claude (via RPi gateway) provides guidance
    ↓ guidance applied
Worker resumes with new context
```

This leverages the existing tiered architecture.

### 1.8 New Files Required

| File | Purpose |
|------|---------|
| `scripts/ralph.py` | Main loop controller |
| `scripts/ralph_verify.py` | Run verification commands, parse results |
| `scripts/ralph_context.py` | Gather git state, build iteration context |
| `shared/ralph/` | Ralph project storage |

---

## Part 2: Adding Agent Teams Capability

### 2.1 Team Architecture

Build on existing brain/worker, add team coordination layer:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TEAM COORDINATOR                               │
│                      scripts/team.py                                │
├─────────────────────────────────────────────────────────────────────┤
│  • Spawns team with role assignments                                │
│  • Manages shared task list                                         │
│  • Routes inter-agent messages                                      │
│  • Synthesizes results                                              │
│  • Enforces file ownership                                          │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   AGENT: lead   │ │ AGENT: coder    │ │ AGENT: reviewer │
│   (brain)       │ │ (worker-1)      │ │ (worker-2)      │
│                 │ │                 │ │                 │
│ Coordinates     │ │ Writes code     │ │ Reviews code    │
│ Plans work      │ │ Owns: src/      │ │ Owns: none      │
│ Synthesizes     │ │                 │ │ Read-only       │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 2.2 Team Configuration

```
shared/teams/
├── {team-name}/
│   ├── config.json         # Team definition
│   ├── tasks/
│   │   ├── pending/
│   │   ├── in_progress/
│   │   └── complete/
│   ├── mailbox/
│   │   ├── lead/           # Messages for lead
│   │   ├── coder/          # Messages for coder
│   │   └── reviewer/       # Messages for reviewer
│   └── state.json          # Team status
```

### 2.3 Team Config Format

```json
{
  "team_name": "auth-refactor",
  "created_at": "2026-02-07T10:00:00Z",
  "objective": "Refactor authentication module for JWT support",

  "agents": [
    {
      "name": "lead",
      "role": "coordinator",
      "model": "qwen2.5:14b",
      "gpu": [0, 3],
      "handler": "brain",
      "system_prompt": "You are the team lead. Plan work, assign tasks, review results. Do not write code directly.",
      "file_ownership": []
    },
    {
      "name": "coder",
      "role": "implementer",
      "model": "qwen2.5:7b",
      "gpu": 1,
      "handler": "worker-1",
      "system_prompt": "You are a code implementer. Write clean, tested code. Follow the lead's instructions.",
      "file_ownership": ["src/auth/*", "tests/test_auth*"]
    },
    {
      "name": "reviewer",
      "role": "reviewer",
      "model": "qwen2.5:7b",
      "gpu": 2,
      "handler": "worker-2",
      "system_prompt": "You are a code reviewer. Find bugs, security issues, and style problems. Be thorough but constructive.",
      "file_ownership": []
    },
    {
      "name": "tester",
      "role": "tester",
      "model": "qwen2.5:7b",
      "gpu": 4,
      "handler": "worker-3",
      "system_prompt": "You are a test engineer. Write comprehensive tests. Verify edge cases.",
      "file_ownership": ["tests/*"]
    }
  ],

  "workflow": {
    "type": "sequential",
    "stages": [
      {"agent": "lead", "action": "plan"},
      {"agent": "coder", "action": "implement"},
      {"agent": "reviewer", "action": "review"},
      {"agent": "coder", "action": "revise"},
      {"agent": "tester", "action": "test"},
      {"agent": "lead", "action": "synthesize"}
    ]
  }
}
```

### 2.4 Inter-Agent Messaging

Add mailbox system to existing task flow:

```python
# Message format
{
  "id": "msg-uuid",
  "from": "reviewer",
  "to": "coder",
  "type": "review_feedback",  # task_assignment, question, finding, etc.
  "content": "Found potential SQL injection in line 42...",
  "relates_to_task": "task-uuid",
  "timestamp": "2026-02-07T10:15:00Z"
}
```

Messages are files in `mailbox/{agent}/` - agents poll their mailbox along with the task queue.

### 2.5 File Ownership Enforcement

Prevent conflicts by tracking file ownership in team coordinator:

```python
def check_file_access(agent_name: str, file_path: str, access_type: str) -> bool:
    """Check if agent can access file."""
    agent = self.agents[agent_name]
    ownership = agent.get("file_ownership", [])

    if access_type == "read":
        return True  # All agents can read

    if access_type == "write":
        for pattern in ownership:
            if fnmatch(file_path, pattern):
                return True
        return False
```

### 2.6 Mapping to Existing Infrastructure

| Agent Teams Concept | Existing Component | Adaptation |
|--------------------|--------------------|------------|
| Lead agent | Brain | Add team context to brain prompts |
| Worker agents | Workers 1-3 | Add role-based system prompts |
| Shared task list | `shared/tasks/` | Add team-specific queue |
| Inter-agent messaging | (new) | Filesystem mailbox |
| File ownership | (new) | Pre-write permission check |
| Team lifecycle | (new) | `team.py` coordinator |

### 2.7 New Files Required

| File | Purpose |
|------|---------|
| `scripts/team.py` | Team coordinator / lifecycle manager |
| `scripts/team_spawn.py` | Create new team from config |
| `scripts/team_message.py` | Send message to agent |
| `scripts/team_status.py` | Show team status dashboard |
| `shared/teams/` | Team storage |

---

## Part 3: Hybrid - Ralph Teams

Combine both: a team of Ralph loops working in parallel.

### 3.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    META-COORDINATOR                                 │
│                    scripts/ralph_team.py                            │
├─────────────────────────────────────────────────────────────────────┤
│  • Divides project into independent file groups                     │
│  • Spawns Ralph loop per file group                                 │
│  • Each loop runs on separate GPU                                   │
│  • Monitors all loops, handles conflicts                            │
│  • Synthesizes when all loops complete                              │
└─────────────────────────────────────────────────────────────────────┘
           │
     ┌─────┴─────┬───────────┐
     ▼           ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ Ralph 1 │ │ Ralph 2 │ │ Ralph 3 │
│ src/api │ │ src/ui  │ │ tests/  │
│ GPU 1   │ │ GPU 2   │ │ GPU 4   │
│ 7B      │ │ 7B      │ │ 7B      │
└─────────┘ └─────────┘ └─────────┘
     │           │           │
     └───────────┴───────────┘
                 │
                 ▼
         Brain (GPUs 0+3)
         Conflict resolution
         Final integration
```

### 3.2 Use Cases

| Scenario | Configuration |
|----------|---------------|
| Large refactor | 3 Ralph loops, each owns different directory |
| Migration | 1 Ralph per framework component |
| Test coverage | 1 Ralph per test file group |
| Multi-module feature | 1 Ralph per module, brain integrates |

### 3.3 Conflict Resolution

When Ralph loops touch overlapping files:

1. **Prevention:** File ownership assigned upfront
2. **Detection:** Git merge conflict on sync
3. **Resolution:** Brain arbitrates, picks winner or merges

---

## Part 4: Implementation Phases

### Phase 1: Ralph Loop (2-3 days)

1. Create `scripts/ralph.py` loop controller
2. Create `scripts/ralph_verify.py` for running verification commands
3. Create `shared/ralph/` directory structure
4. Test with simple refactoring task
5. Add iteration logging to training data

**Deliverable:** Single Ralph loop can run overnight on one worker

### Phase 2: Ralph Escalation (1-2 days)

1. Add stuck detection to ralph.py
2. Add escalation logic (worker → brain)
3. Add Claude escalation via RPi gateway (when available)
4. Test stuck recovery scenarios

**Deliverable:** Ralph can unstick itself by escalating

### Phase 3: Agent Teams (3-4 days)

1. Create `scripts/team.py` coordinator
2. Create mailbox infrastructure
3. Add role-based system prompts to workers
4. Add file ownership checking
5. Test with 3-agent code review scenario

**Deliverable:** Team of 3 agents can review and revise code

### Phase 4: Ralph Teams (2-3 days)

1. Create `scripts/ralph_team.py` meta-coordinator
2. Add file group partitioning logic
3. Add parallel Ralph loop management
4. Add brain-based conflict resolution
5. Test with multi-directory refactor

**Deliverable:** Parallel Ralph loops can refactor large codebase

---

## Part 5: Configuration Updates

### 5.1 config.json Additions

```json
{
  "ralph": {
    "default_max_iterations": 20,
    "default_stuck_threshold": 3,
    "verification_timeout_seconds": 300,
    "context_max_lines": 500,
    "git_commit_each_iteration": true
  },

  "teams": {
    "max_team_size": 4,
    "mailbox_poll_interval_seconds": 2,
    "message_ttl_seconds": 3600,
    "file_ownership_strict": true
  },

  "escalation": {
    "worker_to_brain_after_failures": 3,
    "brain_to_claude_after_failures": 2,
    "claude_gateway_url": "http://rpi-gateway:8080/api"
  }
}
```

### 5.2 New Task Types

| Type | Handler | Purpose |
|------|---------|---------|
| `ralph_iterate` | worker | Single Ralph iteration |
| `ralph_verify` | worker | Run verification commands |
| `team_message` | any | Deliver inter-agent message |
| `team_coordinate` | brain | Team planning/synthesis |

---

## Part 6: Monitoring Enhancements

### 6.1 Ralph Dashboard

```
╔══════════════════════════════════════════════════════════════════╗
║  RALPH LOOP: auth-refactor                      Iteration: 7/20  ║
╠══════════════════════════════════════════════════════════════════╣
║  Status: RUNNING                                                 ║
║  Worker: worker-1 (GPU 1)                                        ║
║                                                                  ║
║  Verification Results:                                           ║
║    [PASS] pytest tests/         (42 tests, 0 failures)           ║
║    [FAIL] mypy src/             (3 errors)                       ║
║    [PASS] ruff check src/                                        ║
║                                                                  ║
║  Current Errors:                                                 ║
║    src/auth/jwt.py:42 - Incompatible return type                 ║
║    src/auth/jwt.py:57 - Missing type annotation                  ║
║    src/auth/session.py:23 - Argument type mismatch               ║
║                                                                  ║
║  Stuck Count: 0/3                                                ║
║  Estimated Completion: 3-5 more iterations                       ║
╚══════════════════════════════════════════════════════════════════╝
```

### 6.2 Team Dashboard

```
╔══════════════════════════════════════════════════════════════════╗
║  TEAM: auth-refactor                                             ║
╠══════════════════════════════════════════════════════════════════╣
║  Agent        │ Role       │ GPU │ Status     │ Current Task     ║
║  ─────────────┼────────────┼─────┼────────────┼────────────────  ║
║  lead         │ coordinator│ 0+3 │ IDLE       │ -                ║
║  coder        │ implementer│ 1   │ WORKING    │ Implement JWT    ║
║  reviewer     │ reviewer   │ 2   │ WAITING    │ -                ║
║  tester       │ tester     │ 4   │ IDLE       │ -                ║
╠══════════════════════════════════════════════════════════════════╣
║  Task Queue: 3 pending, 1 in progress, 5 complete                ║
║  Messages: 2 unread (coder: 1, reviewer: 1)                      ║
║  Files Changed: 4 (src/auth/jwt.py, src/auth/session.py, ...)    ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Part 7: Comparison to Claude Implementation

| Feature | Claude Agent Teams | This Implementation |
|---------|-------------------|---------------------|
| Model | Opus 4.6 (200K+ context) | Qwen 14B + 7B (8-32K context) |
| Parallelism | 10+ agents | 3-4 agents (GPU limited) |
| Inter-agent messaging | Native tool | Filesystem mailbox |
| Context sharing | Full conversation | Disk-based summaries |
| Self-assessment | Reliable | Requires external verification |
| Cost | $$/hour API | Free (local compute) |

### Adaptations for Local Constraints

1. **Shorter context** → Aggressive pruning, disk-based state
2. **Fewer agents** → Prioritize independent work
3. **Weaker instruction following** → Structured output, retries
4. **No reliable self-assessment** → External verification (tests, linter)

---

## Part 8: Quick Start Commands

### Start a Ralph Loop

```bash
# Create Ralph project
mkdir -p shared/ralph/my-refactor
cat > shared/ralph/my-refactor/PROMPT.md << 'EOF'
# Ralph Task: Convert callbacks to async/await

## Objective
Convert all callback-based functions in src/api/ to async/await

## Working Directory
/home/bryan/projects/myapp

## Files to Modify
- src/api/*.py

## Constraints
- Keep all existing tests passing
- Don't change function signatures
EOF

# Start the loop
python scripts/ralph.py my-refactor
```

### Start an Agent Team

```bash
# Create team config
python scripts/team_spawn.py \
  --name code-review \
  --objective "Review and improve src/auth/ security" \
  --agents lead:brain,reviewer:worker-1,security:worker-2

# Monitor the team
python scripts/team_status.py code-review --watch
```

### Start Parallel Ralph Loops

```bash
# Partition project and run parallel loops
python scripts/ralph_team.py \
  --project /home/bryan/projects/myapp \
  --partitions "src/api:api-refactor,src/ui:ui-refactor,tests:test-coverage" \
  --max-iterations 15
```

---

## Part 9: Git Gateway (RPi)

The RPi serves as a **git gateway**, giving LLMs controlled access to version control while preventing them from breaking production code.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL WORLD                              │
│                    (GitHub, GitLab, public repos)                   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ SSH/HTTPS (RPi only)
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      RPi GIT GATEWAY                                │
│                                                                     │
│  • Holds git credentials (SSH keys, tokens)                         │
│  • Clones/pulls from remotes                                        │
│  • Pushes LLM branches for PR review                                │
│  • Human approves PRs → merge to main                               │
│                                                                     │
│  Repos:                                                             │
│    llm_orchestration/          (main project)                       │
│    └── shared/plans/{name}/    (each plan = subproject/branch)      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ shared filesystem
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      GPU RIG (air-gapped)                           │
│                                                                     │
│  • LLMs commit freely to work branches                              │
│  • Cannot push directly (no network)                                │
│  • RPi syncs changes to remote as PRs                               │
└─────────────────────────────────────────────────────────────────────┘
```

### Git Identity for LLMs

Each agent type gets its own git identity for attribution:

```bash
# Ralph loop commits
git config user.name "ralph-worker"
git config user.email "ralph@llm-orchestration.local"

# Agent team commits
git config user.name "agent-coder"
git config user.email "agent-coder@llm-orchestration.local"

# Brain commits (aggregation, planning)
git config user.name "brain-coordinator"
git config user.email "brain@llm-orchestration.local"
```

### Branch Strategy

| Branch Pattern | Purpose | Who Can Push | Merge Policy |
|----------------|---------|--------------|--------------|
| `main` | Production code | Human only | PR + review |
| `llm/ralph/{task}` | Ralph loop iterations | LLM agents | Auto-push, PR to main |
| `llm/team/{name}` | Agent team work | LLM agents | Auto-push, PR to main |
| `llm/experiment/*` | Exploratory work | LLM agents | May never merge |

### Workflow: LLM → PR → Human Review → Main

```
1. Human creates plan, assigns to LLM
2. RPi creates branch: llm/ralph/auth-refactor
3. GPU rig LLMs work, commit to branch (via shared fs)
4. RPi periodically pushes branch to remote
5. When LLM marks complete, RPi opens PR
6. Human reviews PR:
   - Approve → merge to main
   - Request changes → LLM continues work
   - Reject → branch archived
7. Main branch always contains human-approved code
```

### Benefits

| Benefit | Description |
|---------|-------------|
| **Safety** | LLMs cannot break main branch |
| **Attribution** | Git blame shows which agent made changes |
| **Rollback** | Easy revert to any iteration |
| **Audit trail** | Full history of LLM reasoning via commits |
| **Evolution** | LLMs can improve codebase over time |
| **Public repo access** | RPi can clone public repos for LLMs to analyze |

### Reading External Repos

The RPi can clone public repositories for LLMs to study:

```bash
# RPi clones a public repo for LLM reference
git clone https://github.com/example/library.git shared/external/library

# GPU rig LLMs can read it (read-only)
# Useful for: learning patterns, understanding APIs, migration guides
```

LLMs get **read access** to external code without network access.

### Commit Message Convention

```
ralph: iteration 7 - fixed type errors in jwt.py

- Resolved 3 mypy errors in src/auth/jwt.py
- Tests now passing: 42/42

Verification: pytest ✓, mypy ✓, ruff ✓
Iteration: 7/20
Worker: worker-1 (GPU 1)
```

### How This Helps Ralph Loops

| Ralph Need | Git Solution |
|------------|--------------|
| Iteration state | Each iteration = 1 commit |
| Rollback on failure | `git reset --hard HEAD~1` |
| Context between iterations | `git diff HEAD~1` shows what changed |
| Stuck detection | Same diff 3x = stuck |
| Success checkpoint | Tag passing iterations |

### How This Helps Agent Teams

| Team Need | Git Solution |
|-----------|--------------|
| Agent attribution | Commit author = agent identity |
| File ownership | Branch protection rules |
| Conflict detection | Git merge conflicts |
| Review workflow | PR comments for inter-agent feedback |
| Final integration | Brain merges agent branches |

### Config Additions

```json
{
  "git_gateway": {
    "enabled": true,
    "rpi_host": "rpi-gateway",
    "branch_prefix": "llm/",
    "auto_push_interval_seconds": 300,
    "create_pr_on_complete": true,
    "commit_each_iteration": true,
    "llm_identities": {
      "ralph": {"name": "ralph-worker", "email": "ralph@llm-orchestration.local"},
      "brain": {"name": "brain-coordinator", "email": "brain@llm-orchestration.local"},
      "worker-1": {"name": "agent-worker-1", "email": "worker-1@llm-orchestration.local"},
      "worker-2": {"name": "agent-worker-2", "email": "worker-2@llm-orchestration.local"},
      "worker-3": {"name": "agent-worker-3", "email": "worker-3@llm-orchestration.local"}
    }
  }
}
```

### Implementation Phase

Add to Phase 2 (Ralph Escalation):

6. Set up git identities for LLM agents
7. Create branch management scripts on RPi
8. Add commit hooks to ralph.py iterations
9. Test PR workflow with simple task

**Deliverable:** Ralph commits appear as PRs for human review

---

## Summary

The existing llm_orchestration system provides a solid foundation:

| Existing | Enables |
|----------|---------|
| Brain/worker hierarchy | Tiered escalation for both Ralph and Teams |
| Filesystem task queue | Disk-based state for Ralph iterations |
| Dependency tracking | Task sequencing for both paradigms |
| Resource manager design | GPU allocation for parallel loops/agents |
| Training data logging | Continuous improvement of local models |
| RPi gateway (planned) | Git gateway for safe LLM evolution |

**Estimated effort:** 8-12 days for full implementation

**Recommended order:**
1. Ralph Loop (highest ROI, simplest)
2. Ralph Escalation (safety net)
3. Agent Teams (multi-perspective work)
4. Ralph Teams (maximum parallelism)

---

*Document created: 2026-02-07*
*Based on: ralphteamscompare.md, ralphteamslocal.md, existing llm_orchestration architecture*
