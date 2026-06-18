OSS_CAD_SUITE ?= $(HOME)/.local/oss-cad-suite
OSS_BIN := $(OSS_CAD_SUITE)/bin

SIM ?= $(if $(wildcard $(OSS_BIN)/iverilog),$(OSS_BIN)/iverilog,iverilog)
VVP ?= $(if $(wildcard $(OSS_BIN)/vvp),$(OSS_BIN)/vvp,vvp)
YOSYS ?= $(if $(wildcard $(OSS_BIN)/yosys),$(OSS_BIN)/yosys,yosys)
BUILD_DIR ?= build
YOSYS_TOP ?= kalman_filter_axi
YOSYS_LOG ?= $(BUILD_DIR)/yosys-check.log
ARTIX7_PART ?= xc7a35tcsg324-1
VIVADO ?= vivado

RTL = rtl/fixed_mul.v rtl/fixed_div_iter.v rtl/kalman_filter_matrix.v rtl/kalman_filter_axi.v
TB = tb/tb_kalman_filter_matrix.v
AXI_TB = tb/tb_kalman_filter_axi.v

.PHONY: sim sim-core sim-axi yosys-check yosys-artix7 vivado-artix7 vivado-timesim clean

sim: sim-core sim-axi

sim-core: $(BUILD_DIR)/tb_kalman_filter_matrix.vvp
	$(VVP) $<

sim-axi: $(BUILD_DIR)/tb_kalman_filter_axi.vvp
	$(VVP) $<

$(BUILD_DIR)/tb_kalman_filter_matrix.vvp: $(RTL) $(TB)
	mkdir -p $(BUILD_DIR)
	$(SIM) -g2001 -Wall -Irtl -o $@ $(TB) $(RTL)

$(BUILD_DIR)/tb_kalman_filter_axi.vvp: $(RTL) $(AXI_TB)
	mkdir -p $(BUILD_DIR)
	$(SIM) -g2001 -Wall -Irtl -o $@ $(AXI_TB) $(RTL)

yosys-check: $(RTL)
	mkdir -p $(BUILD_DIR)
	$(YOSYS) -q -l $(YOSYS_LOG) -p 'read_verilog $(RTL); hierarchy -top $(YOSYS_TOP); proc; opt; stat'
	@echo "Yosys check complete: $(YOSYS_LOG)"

yosys-artix7: $(RTL) scripts/yosys_artix7.ys
	mkdir -p $(BUILD_DIR)
	$(YOSYS) -q -l $(BUILD_DIR)/yosys-artix7.log scripts/yosys_artix7.ys
	@echo "Yosys Artix-7 synthesis complete: $(BUILD_DIR)/kalman_filter_axi_artix7_synth.v"

vivado-artix7: $(RTL) constraints/artix7_timing.xdc scripts/vivado_artix7.tcl
	ARTIX7_PART=$(ARTIX7_PART) TOP=$(YOSYS_TOP) $(VIVADO) -mode batch -source scripts/vivado_artix7.tcl

vivado-timesim: scripts/vivado_post_route_xsim.tcl
	$(VIVADO) -mode batch -source scripts/vivado_post_route_xsim.tcl

clean:
	rm -rf $(BUILD_DIR)
