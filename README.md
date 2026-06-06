# flux-vm-dispatch

Miniature Flux bytecode VM producing GPU command dispatches. Tests flux-core to cudaclaw synergy.

## Why This Matters

# flux-vm-dispatch
Miniature Flux bytecode VM that interprets ops and produces GPU command dispatches.
Tests how flux-core's bytecode would actually drive cudaclaw's command queue.

## The Five-Layer Stack

This crate is part of the **Oxide Stack** — a distributed GPU runtime built on five layers:

```
┌─────────────────┐
│  cudaclaw        │  Persistent GPU kernels, warp consensus, SmartCRDT
├─────────────────┤
│  cuda-oxide      │  Flux → MIR → Pliron → NVVM → PTX compiler
├─────────────────┤
│  flux-core       │  Bytecode VM + A2A agent protocol
├─────────────────┤
│  pincher         │  "Vector DB as runtime, LLM as compiler"
├─────────────────┤
│  open-parallel   │  Async runtime (tokio fork)
└─────────────────┘
```

The key insight: **ternary values {-1, 0, +1} map directly to GPU compute**. They pack 16× denser than FP32, enable XNOR+popcount matmul, and conservation laws become compile-time checks.

## Design

Every value in this crate follows **ternary algebra** (Z₃):

| Value | Meaning | GPU Analog |
|-------|---------|------------|
| +1 | Positive / Active / Healthy | Warp vote yes |
| 0 | Neutral / Pending / Balanced | Warp vote abstain |
| -1 | Negative / Failed / Overloaded | Warp vote no |

This isn't arbitrary — ternary is the natural encoding for:
1. **BitNet b1.58** (Microsoft) — ternary LLMs at 60% less power
2. **GPU warp voting** — hardware ballot returns ternary consensus
3. **Conservation laws** — {-1, 0, +1} preserves quantity

## Key Types

```rust
pub enum Op
pub enum GpuCmd
pub struct VmState
pub struct FluxVm
pub fn new
pub fn load_program
pub fn step
pub fn run
pub fn state
pub fn trace
pub fn ops_executed
pub fn dispatch_count
```

## Usage

```toml
[dependencies]
flux-vm-dispatch = "0.1.0"
```

```rust
use flux_vm_dispatch::*;
// See src/lib.rs tests for complete working examples
```

## Testing

```bash
git clone https://github.com/SuperInstance/flux-vm-dispatch.git
cd flux-vm-dispatch
cargo test    # 7 tests
```

## Stats

| Metric | Value |
|--------|-------|
| Tests | 7 |
| Lines of Rust | 237 |
| Public API | 12 items |

## License

Apache-2.0
