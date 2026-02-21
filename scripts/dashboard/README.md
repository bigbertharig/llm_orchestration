# Dashboard Package

Web dashboard for monitoring and controlling the LLM orchestration system.

## Quick Start

```bash
# Run from scripts directory
cd /home/bryan/llm_orchestration/scripts
python -m dashboard --port 8787

# Or use the start script
/home/bryan/llm_orchestration/shared/scripts/start_dashboard.sh
```

Access at: http://localhost:8787

## Package Structure

```
dashboard/
├── __init__.py       # Package exports (main, summarize, __version__)
├── __main__.py       # Entry point for `python -m dashboard`
├── server.py         # HTTP handler, template loading, main()
├── data.py           # Task loading, summarize() aggregator
├── workers.py        # Worker/GPU status, telemetry, brain state
├── plans.py          # Plan discovery, batch management, config
├── alerts.py         # Alert collection (failures, completions)
├── chains.py         # Batch chain visualization
├── utils.py          # Constants, utilities, sanitization
└── templates/
    ├── base.css      # Shared CSS variables and styles
    ├── dashboard.html# Main dashboard page structure
    ├── dashboard.js  # Dashboard client-side logic (~800 lines)
    ├── controls.html # Controls page structure
    └── controls.js   # Controls client-side logic (~350 lines)
```

## Module Responsibilities

### utils.py
Pure utility functions with no internal dependencies:
- **Constants**: `HEARTBEAT_WARN_S`, `HEARTBEAT_BAD_S`, `HEARTBEAT_MAX_S`, regex patterns
- **File I/O**: `load_json()`, `load_config()`, `resolve_shared_path()`, `iter_task_files()`
- **Time**: `heartbeat_age_seconds()`, `parse_iso_datetime()`, `format_duration_short()`, `file_mtime_iso()`
- **Sanitization**: `sanitize_text()`, `sanitize_config_value()`, `sanitize_config_object()`, `normalize_github_url()`

### workers.py
Worker and GPU status loading:
- `classify_thermal_cause()` - CPU/GPU/mixed thermal detection
- `load_worker_rows()` - GPU and CPU worker heartbeats
- `load_gpu_telemetry()` - Live nvidia-smi data (local or via SSH)
- `load_brain_state()`, `load_brain_heartbeat()` - Brain agent state

### plans.py
Plan discovery and batch management:
- `shoulders_dir()`, `shoulder_plan_dir()` - Path helpers
- `discover_plans()`, `discover_plan_starters()` - Available plans
- `find_batch_dir()`, `discover_recent_batches()` - Batch history
- `collect_batch_outputs()` - Output file enumeration
- `default_plan_config()`, `plan_input_help()` - Config helpers
- `write_inline_input_file()` - Dashboard-submitted content
- `run_shell()` - Shell command execution

### alerts.py
Alert collection:
- `collect_recent_batch_failure_alerts()` - Recent fatal errors
- `collect_recent_batch_completion_alerts()` - Recent completions

### chains.py
Batch chain building:
- `extract_stage_item()` - Parse task names like `stage_contact_0019`
- `build_batch_chain()` - Dependency chain visualization data

### data.py
Task loading and main aggregator:
- `task_sort_key()`, `to_task_view()`, `lane_view()` - View transforms
- `list_tasks()`, `list_private_tasks()` - Task enumeration
- `count_by_batch()`, `is_system_meta_task()` - Helpers
- `summarize()` - Main data aggregation (imports from all modules)

### server.py
HTTP server and handler:
- `DashboardHandler` - Request routing, API endpoints
- Template loading and caching
- Control actions (kill, resume, start plans)
- `main()` - Server entry point

## API Endpoints

### GET Endpoints
- `/` - Main dashboard HTML
- `/controls` - Controls page HTML
- `/api/status` - JSON status (2-second poll)
- `/api/control/options` - Available plans/batches

### POST Endpoints
- `/api/control/kill_plan` - Kill a batch
- `/api/control/kill_all_active` - Kill all batches
- `/api/control/return_default` - Reset to startup defaults
- `/api/control/resume_plan` - Resume a batch
- `/api/control/start_plan` - Submit a new plan
- `/api/control/batch_outputs` - Get batch output files

## Templates

Templates use `{{PLACEHOLDER}}` syntax for inlining at serve time:
- `{{BASE_CSS}}` - Shared CSS from `base.css`
- `{{DASHBOARD_JS}}` - Dashboard JavaScript
- `{{CONTROLS_JS}}` - Controls JavaScript

CSS/JS are inlined into HTML at serve time (no separate static file serving).

## Configuration

Default config path: `../shared/agents/config.json`

Override with `--config /path/to/config.json`

## Development

The package uses relative imports. Always run from the parent directory:

```bash
cd /home/bryan/llm_orchestration/scripts
python -m dashboard --port 8787
```

To test imports:
```python
from dashboard import main, summarize
from dashboard.utils import load_json
from dashboard.workers import load_worker_rows
```
