set project_root [file normalize [file join [file dirname [info script]] ..]]
set sim_dir [file join $project_root build vivado_artix7_xsim]
set impl_dir [file join $project_root build vivado_artix7]

file mkdir $sim_dir
cd $sim_dir

set timesim_netlist [file join $impl_dir kalman_filter_axi_post_route_timesim.v]
set sdf_file [file join $impl_dir kalman_filter_axi_post_route.sdf]

if {![file exists $timesim_netlist]} {
    error "Missing timing netlist: $timesim_netlist. Run make vivado-artix7 first."
}
if {![file exists $sdf_file]} {
    error "Missing SDF: $sdf_file. Run make vivado-artix7 first."
}

xvlog $timesim_netlist
xvlog [file join $project_root tb tb_kalman_filter_axi.v]
xelab tb_kalman_filter_axi -s tb_kalman_filter_axi_timesim -timescale 1ns/1ps -sdfmax /tb_kalman_filter_axi/dut=$sdf_file
xsim tb_kalman_filter_axi_timesim -runall
