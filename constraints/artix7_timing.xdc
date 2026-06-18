## Timing-only out-of-context constraints for the Kalman AXI wrapper on
## Xilinx Artix-7.
## Default target used by scripts: xc7a35tcsg324-1.
##
## The AXI/Stream ports are intended to connect to internal FPGA interconnect,
## not directly to package pins. Therefore this file intentionally avoids
## PACKAGE_PIN and IOSTANDARD constraints. Add board-level pin constraints in a
## separate top-level project if you wrap this IP with physical IO.

create_clock -name aclk -period 10.000 [get_ports aclk]

set_false_path -from [get_ports aresetn]

set input_ports [remove_from_collection [all_inputs] [get_ports {aclk aresetn}]]
set output_ports [all_outputs]

set_input_delay -clock [get_clocks aclk] -max 2.000 $input_ports
set_input_delay -clock [get_clocks aclk] -min 0.500 $input_ports

set_output_delay -clock [get_clocks aclk] -max 2.000 $output_ports
set_output_delay -clock [get_clocks aclk] -min 0.500 $output_ports

set_clock_uncertainty 0.200 [get_clocks aclk]
