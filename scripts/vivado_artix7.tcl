set project_root [file normalize [file join [file dirname [info script]] ..]]
set build_dir [file join $project_root build vivado_artix7]
set reports_dir [file join $project_root reports]

file mkdir $build_dir
file mkdir $reports_dir

set part_name [expr {[info exists ::env(ARTIX7_PART)] ? $::env(ARTIX7_PART) : "xc7a35tcsg324-1"}]
set top_name [expr {[info exists ::env(TOP)] ? $::env(TOP) : "kalman_filter_axi"}]

create_project -force kalman_filter_artix7 $build_dir -part $part_name

read_verilog [file join $project_root rtl fixed_mul.v]
read_verilog [file join $project_root rtl fixed_div_iter.v]
read_verilog [file join $project_root rtl kalman_filter_matrix.v]
read_verilog [file join $project_root rtl kalman_filter_axi.v]
read_xdc [file join $project_root constraints artix7_timing.xdc]

synth_design -top $top_name -part $part_name -mode out_of_context
write_checkpoint -force [file join $build_dir post_synth.dcp]
report_utilization -file [file join $reports_dir vivado_post_synth_utilization.rpt]
report_timing_summary -file [file join $reports_dir vivado_post_synth_timing_summary.rpt]

opt_design
place_design
phys_opt_design
route_design

write_checkpoint -force [file join $build_dir post_route.dcp]
report_utilization -file [file join $reports_dir vivado_post_route_utilization.rpt]
report_timing_summary -file [file join $reports_dir vivado_post_route_timing_summary.rpt]

write_verilog -force -mode timesim -sdf_anno true [file join $build_dir kalman_filter_axi_post_route_timesim.v]
write_sdf -force [file join $build_dir kalman_filter_axi_post_route.sdf]

puts "Vivado Artix-7 implementation complete."
puts "Part: $part_name"
puts "Mode: out-of-context IP implementation"
puts "Post-route timing report: [file join $reports_dir vivado_post_route_timing_summary.rpt]"
puts "Timing simulation netlist: [file join $build_dir kalman_filter_axi_post_route_timesim.v]"
puts "SDF: [file join $build_dir kalman_filter_axi_post_route.sdf]"
