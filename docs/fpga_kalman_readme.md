# FPGA Kalman Filter Verilog

This folder adds a synthesizable Verilog-2001 Kalman filter IP core using signed fixed-point Q16.16 arithmetic. The default configuration targets a 4-state, 2-measurement model with a 2x2 analytic inverse for the measurement covariance matrix.

## Files

- `rtl/kalman_filter_matrix.v`: reusable Kalman filter core with `valid/ready` handshake, register-style configuration, and a counter-driven MAC datapath.
- `rtl/kalman_filter_axi.v`: AXI-Lite configuration and AXI-Stream input/output wrapper around the reusable core.
- `rtl/fixed_mul.v`: signed fixed-point multiply with saturation.
- `rtl/fixed_div_iter.v`: multi-cycle restoring divider for the determinant reciprocal.
- `tb/tb_kalman_filter_matrix.v`: core smoke test.
- `tb/tb_kalman_filter_axi.v`: AXI-Lite/AXI-Stream wrapper smoke test.
- `constraints/artix7_timing.xdc`: timing-only out-of-context Artix-7 constraints.
- `scripts/yosys_artix7.ys`: Yosys Xilinx 7-series synthesis script.
- `scripts/vivado_artix7.tcl`: Vivado OOC implementation and post-route SDF/netlist export script.
- `scripts/vivado_post_route_xsim.tcl`: XSim post-route timing simulation script.

## Default Model

- State: `[x, vx, y, vy]`
- Measurement: `[x_meas, y_meas]`
- `F = [[1,1,0,0], [0,1,0,0], [0,0,1,1], [0,0,0,1]]`
- `H = [[1,0,0,0], [0,0,1,0]]`

Matrix products are scheduled as one dot-product term per cycle with explicit `row/col/inner` counters, so the design maps to FPGA arithmetic datapaths instead of unrolling full matrix equations into one large combinational block.

## Core Configuration Address Map

All matrices are row-major Q16.16 signed values. Writes are accepted only while the core is idle.

- `0x00..0x0F`: `F`, 16 words
- `0x20..0x27`: `H`, 8 words
- `0x40..0x4F`: `Q`, 16 words
- `0x60..0x63`: `R`, 4 words
- `0x80..0x83`: state `x`, 4 words
- `0xA0..0xAF`: covariance `P`, 16 words

## AXI Wrapper Address Map

The AXI-Lite wrapper uses byte addresses. The matrix register byte address is the core word address multiplied by four.

- `0x000..0x03C`: `F`
- `0x080..0x09C`: `H`
- `0x100..0x13C`: `Q`
- `0x180..0x18C`: `R`
- `0x200..0x20C`: state `x`
- `0x280..0x2BC`: covariance `P`
- `0x300`: status `{err_singular, out_valid, in_ready, busy}`
- `0x304`: version `0x4B465001`
- `0x308`: packed parameters `{N, M, DATA_WIDTH}`

AXI-Stream input `s_axis_z_tdata` carries one packed measurement vector. AXI-Stream output `m_axis_x_tdata` carries one packed state vector, and `m_axis_x_tuser[0]` reports `err_singular`.

## Simulation And Synthesis

Run the RTL simulations:

```sh
make sim
```

Run the Yosys frontend/synthesis check:

```sh
make yosys-check
```

Run Yosys Xilinx 7-series synthesis:

```sh
make yosys-artix7
```

Run Vivado implementation and post-route timing netlist generation on a machine with Vivado installed:

```sh
make vivado-artix7
```

The default target part is `xc7a35tcsg324-1`. Override it with:

```sh
make vivado-artix7 ARTIX7_PART=xc7a100tcsg324-1
```

After `make vivado-artix7`, run post-route timing simulation with XSim:

```sh
make vivado-timesim
```
