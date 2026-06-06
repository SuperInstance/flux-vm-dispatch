//! # flux-vm-dispatch
//!
//! Miniature Flux bytecode VM that interprets ops and produces GPU command dispatches.
//! Tests how flux-core's bytecode would actually drive cudaclaw's command queue.

use std::collections::VecDeque;

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    MOVI { reg: u8, imm: i16 },
    ADD { rd: u8, rs1: u8, rs2: u8 },
    SUB { rd: u8, rs1: u8, rs2: u8 },
    TADD { rd: u8, ra: u8, rb: u8 },
    TMUL { rd: u8, ra: u8, rb: u8 },
    SYNC,
    LOAD { rd: u8, addr: u8 },
    STORE { addr: u8, rs: u8 },
    HALT,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuCmd {
    KernelLaunch { name: String, threads: u32 },
    BarrierSync,
    MemCopy { src: u8, dst: u8, size: u32 },
}

#[derive(Debug, Clone)]
pub struct VmState {
    pub registers: [i32; 16],
    pub memory: [i32; 256],
    pub pc: usize,
    pub halted: bool,
    pub sync_count: u32,
}

impl Default for VmState {
    fn default() -> Self {
        Self { registers: [0; 16], memory: [0; 256], pc: 0, halted: false, sync_count: 0 }
    }
}

pub struct FluxVm {
    state: VmState,
    dispatch_queue: VecDeque<GpuCmd>,
    trace: Vec<String>,
    ops_executed: u64,
}

impl FluxVm {
    pub fn new() -> Self {
        Self { state: VmState::default(), dispatch_queue: VecDeque::new(), trace: Vec::new(), ops_executed: 0 }
    }

    pub fn load_program(&mut self, ops: &[Op]) {
        // Pre-load memory with simulated data
        for i in 0..16 { self.state.memory[i] = i as i32; }
    }

    pub fn step(&mut self, op: &Op) -> Option<GpuCmd> {
        if self.state.halted { return None; }
        self.ops_executed += 1;
        let cmd = match op {
            Op::MOVI { reg, imm } => {
                self.state.registers[*reg as usize] = *imm as i32;
                self.trace.push(format!("MOVI r{}, {}", reg, imm));
                None
            }
            Op::ADD { rd, rs1, rs2 } => {
                let a = self.state.registers[*rs1 as usize];
                let b = self.state.registers[*rs2 as usize];
                self.state.registers[*rd as usize] = a + b;
                self.trace.push(format!("ADD r{}, r{}, r{}", rd, rs1, rs2));
                Some(GpuCmd::KernelLaunch { name: "iadd".into(), threads: 256 })
            }
            Op::SUB { rd, rs1, rs2 } => {
                let a = self.state.registers[*rs1 as usize];
                let b = self.state.registers[*rs2 as usize];
                self.state.registers[*rd as usize] = a - b;
                Some(GpuCmd::KernelLaunch { name: "isub".into(), threads: 256 })
            }
            Op::TADD { rd, ra, rb } => {
                let a = self.state.registers[*ra as usize];
                let b = self.state.registers[*rb as usize];
                // Z₃ addition
                let r = match (a, b) {
                    (-1, -1) => 1, (-1, 0) => -1, (-1, 1) => 0,
                    (0, -1) => -1, (0, 0) => 0, (0, 1) => 1,
                    (1, -1) => 0, (1, 0) => 1, (1, 1) => -1,
                    _ => 0,
                };
                self.state.registers[*rd as usize] = r;
                self.trace.push(format!("TADD r{}, r{}, r{} = {}", rd, ra, rb, r));
                Some(GpuCmd::KernelLaunch { name: "ternary_add".into(), threads: 256 })
            }
            Op::TMUL { rd, ra, rb } => {
                let a = self.state.registers[*ra as usize];
                let b = self.state.registers[*rb as usize];
                let r = match (a, b) {
                    (-1, -1) => 1, (-1, 0) => 0, (-1, 1) => -1,
                    (0, -1) => 0, (0, 0) => 0, (0, 1) => 0,
                    (1, -1) => -1, (1, 0) => 0, (1, 1) => 1,
                    _ => 0,
                };
                self.state.registers[*rd as usize] = r;
                self.trace.push(format!("TMUL r{}, r{}, r{} = {}", rd, ra, rb, r));
                Some(GpuCmd::KernelLaunch { name: "ternary_mul".into(), threads: 256 })
            }
            Op::SYNC => {
                self.state.sync_count += 1;
                self.trace.push("SYNC".into());
                Some(GpuCmd::BarrierSync)
            }
            Op::LOAD { rd, addr } => {
                self.state.registers[*rd as usize] = self.state.memory[*addr as usize];
                self.trace.push(format!("LOAD r{}, [{}]", rd, addr));
                Some(GpuCmd::MemCopy { src: *addr, dst: *rd, size: 4 })
            }
            Op::STORE { addr, rs } => {
                self.state.memory[*addr as usize] = self.state.registers[*rs as usize];
                self.trace.push(format!("STORE [{}], r{}", addr, rs));
                Some(GpuCmd::MemCopy { src: *rs, dst: *addr, size: 4 })
            }
            Op::HALT => {
                self.state.halted = true;
                self.trace.push("HALT".into());
                None
            }
        };
        if let Some(ref c) = cmd { self.dispatch_queue.push_back(c.clone()); }
        cmd
    }

    pub fn run(&mut self, program: &[Op]) -> Vec<GpuCmd> {
        self.load_program(program);
        let mut cmds = Vec::new();
        for op in program {
            if self.state.halted { break; }
            if let Some(c) = self.step(op) { cmds.push(c); }
        }
        cmds
    }

    pub fn state(&self) -> &VmState { &self.state }
    pub fn trace(&self) -> &[String] { &self.trace }
    pub fn ops_executed(&self) -> u64 { self.ops_executed }
    pub fn dispatch_count(&self) -> usize { self.dispatch_queue.len() }
}

impl Default for FluxVm {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_movi() {
        let mut vm = FluxVm::new();
        vm.step(&Op::MOVI { reg: 0, imm: 42 });
        assert_eq!(vm.state().registers[0], 42);
    }

    #[test]
    fn test_arithmetic() {
        let mut vm = FluxVm::new();
        vm.step(&Op::MOVI { reg: 0, imm: 10 });
        vm.step(&Op::MOVI { reg: 1, imm: 20 });
        vm.step(&Op::ADD { rd: 2, rs1: 0, rs2: 1 });
        assert_eq!(vm.state().registers[2], 30);
        vm.step(&Op::SUB { rd: 3, rs1: 1, rs2: 0 });
        assert_eq!(vm.state().registers[3], 10);
    }

    #[test]
    fn test_ternary_ops() {
        let mut vm = FluxVm::new();
        vm.step(&Op::MOVI { reg: 0, imm: 1 });
        vm.step(&Op::MOVI { reg: 1, imm: -1 });
        vm.step(&Op::TADD { rd: 2, ra: 0, rb: 1 });
        assert_eq!(vm.state().registers[2], 0); // 1 + (-1) = 0
        vm.step(&Op::TMUL { rd: 3, ra: 0, rb: 1 });
        assert_eq!(vm.state().registers[3], -1); // 1 * (-1) = -1
    }

    #[test]
    fn test_gpu_dispatch() {
        let mut vm = FluxVm::new();
        let cmds = vm.run(&[
            Op::MOVI { reg: 0, imm: 1 },
            Op::ADD { rd: 1, rs1: 0, rs2: 0 },
            Op::SYNC,
            Op::HALT,
        ]);
        assert_eq!(cmds.len(), 2); // ADD dispatches, SYNC dispatches
        assert_eq!(cmds[0], GpuCmd::KernelLaunch { name: "iadd".into(), threads: 256 });
        assert_eq!(cmds[1], GpuCmd::BarrierSync);
    }

    #[test]
    fn test_load_store() {
        let mut vm = FluxVm::new();
        vm.state.memory[10] = 99;
        vm.step(&Op::LOAD { rd: 5, addr: 10 });
        assert_eq!(vm.state().registers[5], 99);
        vm.step(&Op::STORE { addr: 20, rs: 5 });
        assert_eq!(vm.state().memory[20], 99);
    }

    #[test]
    fn test_full_program() {
        let program = vec![
            Op::MOVI { reg: 0, imm: 1 },
            Op::MOVI { reg: 1, imm: -1 },
            Op::TADD { rd: 2, ra: 0, rb: 1 },
            Op::TMUL { rd: 3, ra: 2, rb: 0 },
            Op::SYNC,
            Op::STORE { addr: 0, rs: 3 },
            Op::HALT,
        ];
        let mut vm = FluxVm::new();
        let cmds = vm.run(&program);
        assert!(vm.state().halted);
        assert!(cmds.len() >= 3); // TADD, TMUL, SYNC, STORE all dispatch
        assert_eq!(vm.state().memory[0], 0); // TADD(1,-1)=0, TMUL(0,1)=0
    }

    #[test]
    fn test_trace() {
        let mut vm = FluxVm::new();
        vm.run(&[Op::MOVI { reg: 0, imm: 5 }, Op::HALT]);
        assert!(vm.trace().len() >= 2);
        assert!(vm.trace()[0].contains("MOVI"));
    }
}
