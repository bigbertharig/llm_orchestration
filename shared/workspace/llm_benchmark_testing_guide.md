# LLM Benchmark Testing Guide
## For Multi-GPU Agent Rig - Model Evaluation & Selection

---

## Purpose
Evaluate candidate LLM models for use in a tiered agent architecture:
- **Brain tier**: Coordination, planning, decision-making (needs high reasoning quality)
- **Worker tier**: Task execution, data processing, instruction following (needs reliability and speed)
- **Cloud escalation**: Frontier-level reasoning via RPi gateway (for tasks exceeding local capability)

---

## Part 1: Standard Public Benchmarks (Pre-Screening)

Use these to narrow the field before downloading and testing locally.

### Key Benchmarks to Check

| Benchmark | What It Tests | Why It Matters |
|-----------|--------------|----------------|
| **MMLU** | Breadth of knowledge across 57 subjects | General intelligence indicator |
| **HumanEval / MBPP** | Code generation (write code that passes unit tests) | Agent needs to write and debug scripts |
| **GSM8K** | Multi-step math word problems | Tests sequential reasoning chains - critical for agent planning |
| **ARC Challenge** | Science questions requiring reasoning | Common sense and causal reasoning |
| **HellaSwag** | Sentence completion / common sense | Understanding real-world consequences of actions |
| **Winogrande** | Pronoun resolution / language understanding | True comprehension vs pattern matching |
| **MT-Bench** | Multi-turn conversation quality | Coherence across agent loop iterations |

### Rough Score Ranges by Model Size
- **7B models**: MMLU 55-65%, GSM8K 40-60%, HumanEval 30-50%
- **13B models**: MMLU 65-75%, GSM8K 60-75%, HumanEval 40-60%
- **70B models**: MMLU 75-85%, GSM8K 80-90%, HumanEval 60-75%
- **Frontier (cloud)**: MMLU 85-90%+, GSM8K 90%+, HumanEval 80%+

### Where to Find Results
- **Hugging Face Open LLM Leaderboard**: Standardized scores for all open source models
- **LMSys Chatbot Arena**: ELO rankings from human side-by-side comparisons
- **Model cards on Hugging Face**: Individual model benchmark results

---

## Part 2: Hardware Performance Benchmarks (Run Locally)

Measure actual performance on the rig before committing to a model.

### Tests to Run

**Tokens Per Second (Generation Speed)**
```bash
# Using llama.cpp - run same prompt through each model
./main -m model.gguf -p "Explain the process of photosynthesis in detail" -n 512 --verbose
# Note the "eval time" and "tokens per second" in output
```

**VRAM Usage Under Load**
```bash
# Monitor during inference with typical prompt length
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv -l 1
# Test with short prompt, medium prompt, and maximum context length
```

**Multi-GPU Layer Split Performance** (for brain cards)
```bash
# Test splitting across 2 GPUs vs single GPU
# Compare tokens/sec at different layer distributions
# Record optimal split ratio for each model
```

**Parallel Instance Throughput** (for worker cards)
```bash
# Run N independent instances simultaneously (one per GPU)
# Measure aggregate tokens/sec
# Find the point where adding instances degrades per-instance performance
```

### Record Template

| Model | Quant | VRAM (idle) | VRAM (peak) | Tokens/sec | GPU Config | Notes |
|-------|-------|-------------|-------------|------------|------------|-------|
| Llama 3.1 8B | Q4_K_M | | | | Single 1060 | |
| Mistral 7B | Q5_K_M | | | | Single 1060 | |
| Llama 3.1 8B | Q6_K | | | | Single 1060 | |
| Phi-3 3.8B | Q8_0 | | | | Single 1060 | |

---

## Part 3: Task-Specific Agent Benchmarks (Run Locally)

These test real-world capability for the agent workloads.

### Test Suite: 10 Core Tasks

Score each model 0-2 per task:
- **0** = Failed or produced harmful/wrong output
- **1** = Partially correct, needed human correction
- **2** = Fully correct, production-ready output

#### Test 1: Parse Structured Data
```
Input: Raw nvidia-smi output (paste actual output from rig)
Prompt: "Parse this nvidia-smi output and return a JSON object with 
fields: gpu_id, name, temperature_c, memory_used_mb, memory_total_mb, 
utilization_pct for each GPU."
Evaluate: Valid JSON? Correct values? Correct field names?
```

#### Test 2: Follow Multi-Step Instructions
```
Prompt: "Do the following in order:
1. List all files in /shared/tasks/queue/
2. For each .json file, read the 'priority' field
3. Sort them by priority (highest first)
4. Write the sorted list to /shared/tasks/priority_order.txt
5. Report how many tasks are queued
Write the bash commands to accomplish this."
Evaluate: Correct order? All steps addressed? Commands would actually work?
```

#### Test 3: Write a Bash Command
```
Prompt: "Write a single bash command that finds all PDF files in /shared/downloads/ 
larger than 10MB, modified in the last 7 days, and copies them to /shared/processing/"
Evaluate: Syntactically correct? Would it actually work? Edge cases handled?
```

#### Test 4: Summarize Concisely
```
Input: A 500-word technical document
Prompt: "Summarize this in exactly 3 bullet points, under 50 words total."
Evaluate: Under word limit? Captured key points? Followed format?
```

#### Test 5: Diagnose an Error
```
Input: A Python traceback or bash error output
Prompt: "What went wrong and how do I fix it?"
Evaluate: Correct diagnosis? Fix would work? Didn't hallucinate a cause?
```

#### Test 6: Make a Tradeoff Decision
```
Prompt: "I can run Model A (13B, Q4, 25 tokens/sec) or Model B (7B, Q6, 45 tokens/sec) 
for a task that involves parsing 500 log files and extracting error patterns. 
Which should I use and why?"
Evaluate: Reasonable recommendation? Considered relevant factors? Clear reasoning?
```

#### Test 7: Output in Specific Schema
```
Prompt: "Generate a task file in this exact format:
{
  'task_id': string,
  'action': string,
  'target': string,
  'priority': int 1-5,
  'estimated_duration_minutes': int,
  'requires_gpu': boolean
}
The task is: embed all PDF files in the florida-codes directory."
Evaluate: Valid JSON? All fields present? Correct types? Reasonable values?
```

#### Test 8: Handle Ambiguity
```
Prompt: "Process the new files."
Evaluate: Does it ask for clarification? Does it make reasonable assumptions 
and state them? Or does it just guess and proceed?
```

#### Test 9: Know Its Limits
```
Prompt: "What is the current price of Bitcoin?"
Evaluate: Does it say it doesn't have real-time data? Or does it hallucinate a number?

Prompt: "Is this bash command safe to run? rm -rf /tmp/old_*"
Evaluate: Does it flag the potential risk of glob expansion? Or just say yes?
```

#### Test 10: Multi-Step Reasoning
```
Prompt: "GPU 0 has 2GB free VRAM. GPU 1 has 4GB free. GPU 2 has 6GB free. 
I need to run an embedding model (1.5GB) and a 7B inference model (4.5GB). 
Which GPUs should I assign each to, and why?"
Evaluate: Correct allocation? Sound reasoning? Considered constraints?
```

### Scoring Summary Template

| Model | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | Total (/20) | tok/s | Quality-Adjusted Score |
|-------|----|----|----|----|----|----|----|----|----|----|-------------|-------|----------------------|
| | | | | | | | | | | | | | Total × tok/s |

---

## Part 4: Tier Assignment Criteria

### Brain Tier Requirements
- Total score: 16+ /20
- Must score 2 on: Test 6 (tradeoffs), Test 8 (ambiguity), Test 10 (reasoning)
- GSM8K public benchmark: 75%+
- Speed secondary to quality

### Worker Tier Requirements
- Total score: 12+ /20
- Must score 2 on: Test 1 (parsing), Test 3 (bash), Test 7 (schema)
- Speed weighted heavily — quality-adjusted score matters more than raw quality
- Consistency more important than peak performance

### Cloud Escalation Triggers
- Task requires capabilities the brain scored 0-1 on
- Task involves ambiguity the brain can't resolve
- Task needs current/real-time information
- Task involves risk where a wrong decision has significant consequences

---

## Part 5: Automated Benchmark Pipeline (Future)

Once the agent system is operational, automate model evaluation:

1. New model appears on Hugging Face matching hardware requirements
2. Brain agent tasks a worker: "Download and quantize this model"
3. Brain runs the benchmark suite against the new model
4. Results written to /shared/knowledge/model_benchmarks/
5. Brain compares against current models and recommends whether to adopt

This creates a self-improving system that automatically evaluates new models 
as the open source ecosystem evolves.

---

## Notes
- Re-run benchmarks when upgrading hardware (new GPU changes the speed calculus)
- Re-run benchmarks when quantization tools improve (better quants = better quality at same size)
- Keep all benchmark results on the shared drive for historical comparison
- Test at the quantization level you'll actually run, not at full precision
