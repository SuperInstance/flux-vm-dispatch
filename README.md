# flux-vm-dispatch: GPU Command Dispatch from Flux Bytecode

A miniature virtual machine that interprets Flux bytecode instructions and translates them into GPU command dispatches. It bridges `flux-core`'s bytecode semantics with the `cudaclaw` command queue, testing the full pipeline from high-level agent instructions down to GPU kernel launches.

## Why It Matters

The gap between "an agent decides to compute something" and "a GPU actually runs the kernel" is where many frameworks break down. This crate proves the interface is sound by implementing a complete dispatch loop: every arithmetic or memory operation in the Flux ISA maps to a concrete GPU command. It also introduces **balanced ternary** arithmetic (Z₃), enabling exploration of ternary computing on binary GPU hardware.

## How It Works

### Instruction Set

The VM supports a compact instruction set with both binary and ternary operations:

| Instruction | Operands | GPU Dispatch | Semantics |
|-------------|----------|-------------|-----------|
| `MOVI` | reg, imm | — | Load immediate |
| `ADD` | rd, rs1, rs2 | `KernelLaunch("iadd", 256)` | rd = rs1 + rs2 |
| `SUB` | rd, rs1, rs2 | `KernelLaunch("isub", 256)` | rd = rs1 − rs2 |
| `TADD` | rd, ra, rb | `KernelLaunch("ternary_add", 256)` | Z₃ addition |
| `TMUL` | rd, ra, rb | `KernelLaunch("ternary_mul", 256)` | Z₃ multiplication |
| `SYNC` | — | `BarrierSync` | Global synchronization |
| `LOAD` | rd, addr | `MemCopy(addr → rd, 4B)` | Memory load |
| `STORE` | addr, rs | `MemCopy(rs → addr, 4B)` | Memory store |
| `HALT` | — | — | Stop execution |

### Balanced Ternary Arithmetic (Z₃)

The ternary operations implement arithmetic over Z₃ = {−1, 0, +1}, the **balanced ternary** number system studied by Brusentsov:

**Addition table (TADD):**

| + | −1 | 0 | +1 |
|---|-----|---|-----|
| **−1** | +1 | −1 | 0 |
| **0** | −1 | 0 | +1 |
| **+1** | 0 | +1 | −1 |

**Multiplication table (TMUL):**

| × | −1 | 0 | +1 |
|---|-----|---|-----|
| **−1** | +1 | 0 | −1 |
| **0** | 0 | 0 | 0 |
| **+1** | −1 | 0 | +1 |

This forms a ring isomorphic to ℤ/3ℤ under the mapping {−1 → 2, 0 → 0, +1 → 1}.

### VM State

```
registers: [i32; 16]    // 16 general-purpose registers
memory:    [i32; 256]   // 256-word addressable memory
pc:        usize        // program counter
halted:    bool         // execution flag
sync_count: u32         // barrier operations
```

### Complexity

| Operation | Time | Dispatch |
|-----------|------|----------|
| `MOVI` | O(1) | None |
| `ADD`/`SUB` | O(1) | One kernel launch |
| `TADD`/`TMUL` | O(1) | One kernel launch |
| `LOAD`/`STORE` | O(1) | One MemCopy |
| `SYNC` | O(1) | One BarrierSync |
| Full program (n ops) | O(n) | Up to n GPU commands |

## Quick Start

```rust
use flux_vm_dispatch::{FluxVm, Op, GpuCmd};

let program = vec![
    Op::MOVI { reg: 0, imm: 1 },
    Op::MOVI { reg: 1, imm: -1 },
    Op::TADD { rd: 2, ra: 0, rb: 1 },  // 1 + (-1) = 0 in Z₃
    Op::SYNC,
    Op::HALT,
];

let mut vm = FluxVm::new();
let cmds = vm.run(&program);

assert_eq!(vm.state().registers[2], 0);  // Z₃ result
assert!(cmds.iter().any(|c| matches!(c, GpuCmd::BarrierSync)));
```

## API

### `FluxVm`

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `() -> Self` | Initialize with default state |
| `load_program` | `(&mut self, &[Op])` | Pre-load memory with simulated data |
| `step` | `(&mut self, &Op) -> Option<GpuCmd>` | Execute one instruction, return dispatch |
| `run` | `(&mut self, &[Op]) -> Vec<GpuCmd>` | Execute full program |
| `state` | `(&self) -> &VmState` | Read-only VM state |
| `trace` | `(&self) -> &[String]` | Human-readable execution log |
| `ops_executed` | `(&self) -> u64` | Instruction count |
| `dispatch_count` | `(&self) -> usize` | GPU commands queued |

### `GpuCmd`

```
KernelLaunch { name: String, threads: u32 }
BarrierSync
MemCopy { src: u8, dst: u8, size: u32 }
```

## Architecture Notes

This crate sits at the **γ/η boundary** in the γ + η = C framework. The VM dispatch logic is **γ** — it is deterministic, with fixed opcode-to-GPU-command mappings. The dispatch queue that receives these commands (in `cudaclaw`) is **η** — it reorders, batches, and schedules them. The `TADD`/`TMUL` operations are particularly interesting as they encode **Information–Physics duality**: balanced ternary is the most efficient radix for computation (radix economy e ≈ 2.718), connecting information theory to physical implementation.

## References

- Brusentsov, N. P. (1958). *The Computer "Setun"*. Moscow University Press.
- Knuth, D. E. (1997). *The Art of Computer Programming, Vol. 2* (3rd ed.), §4.1. Addison-Wesley.
- Frieder, G. & Luk, C. (1975). *Ternary Computers*. Proc. ACM Annual Conf.

## License

MIT
