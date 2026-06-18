`timescale 1ns/1ps

module kalman_filter_axi #(
    parameter DATA_WIDTH = 32,
    parameter FRAC_WIDTH = 16,
    parameter N = 4,
    parameter M = 2,
    parameter AXIL_ADDR_WIDTH = 12,
    parameter CFG_ADDR_WIDTH = 8
) (
    input aclk,
    input aresetn,

    input [AXIL_ADDR_WIDTH-1:0] s_axil_awaddr,
    input s_axil_awvalid,
    output s_axil_awready,
    input [DATA_WIDTH-1:0] s_axil_wdata,
    input [(DATA_WIDTH/8)-1:0] s_axil_wstrb,
    input s_axil_wvalid,
    output s_axil_wready,
    output [1:0] s_axil_bresp,
    output s_axil_bvalid,
    input s_axil_bready,

    input [AXIL_ADDR_WIDTH-1:0] s_axil_araddr,
    input s_axil_arvalid,
    output s_axil_arready,
    output [DATA_WIDTH-1:0] s_axil_rdata,
    output [1:0] s_axil_rresp,
    output s_axil_rvalid,
    input s_axil_rready,

    input [M*DATA_WIDTH-1:0] s_axis_z_tdata,
    input s_axis_z_tvalid,
    output s_axis_z_tready,
    input s_axis_z_tlast,

    output [N*DATA_WIDTH-1:0] m_axis_x_tdata,
    output m_axis_x_tvalid,
    input m_axis_x_tready,
    output m_axis_x_tlast,
    output [0:0] m_axis_x_tuser
);
    localparam [1:0] AXI_RESP_OKAY   = 2'b00;
    localparam [1:0] AXI_RESP_SLVERR = 2'b10;

    localparam [CFG_ADDR_WIDTH-1:0] STATUS_WORD  = 8'hC0;
    localparam [CFG_ADDR_WIDTH-1:0] VERSION_WORD = 8'hC1;
    localparam [CFG_ADDR_WIDTH-1:0] PARAM_WORD   = 8'hC2;
    localparam [7:0] N_U8 = N;
    localparam [7:0] M_U8 = M;
    localparam [7:0] DATA_WIDTH_U8 = DATA_WIDTH;

    reg awready_r;
    reg wready_r;
    reg bvalid_r;
    reg [1:0] bresp_r;
    reg arready_r;
    reg rvalid_r;
    reg [1:0] rresp_r;
    reg [DATA_WIDTH-1:0] rdata_r;
    reg aw_hold_valid;
    reg w_hold_valid;
    reg [AXIL_ADDR_WIDTH-1:0] awaddr_hold;
    reg [DATA_WIDTH-1:0] wdata_hold;
    reg [(DATA_WIDTH/8)-1:0] wstrb_hold;

    reg cfg_we_r;
    reg [CFG_ADDR_WIDTH-1:0] cfg_addr_r;
    reg signed [DATA_WIDTH-1:0] cfg_wdata_r;
    wire [CFG_ADDR_WIDTH-1:0] cfg_addr_mux;
    wire signed [DATA_WIDTH-1:0] cfg_rdata;

    wire core_in_ready;
    wire core_out_valid;
    wire core_busy;
    wire core_err_singular;
    wire [N*DATA_WIDTH-1:0] core_x_out;

    wire [CFG_ADDR_WIDTH-1:0] aw_word_addr;
    wire [CFG_ADDR_WIDTH-1:0] held_aw_word_addr;
    wire [CFG_ADDR_WIDTH-1:0] ar_word_addr;
    wire aw_fire;
    wire w_fire;
    wire have_aw;
    wire have_w;
    wire write_is_status_space;
    wire read_is_status;
    wire read_is_version;
    wire read_is_param;
    wire write_full_strobe;

    assign aw_word_addr = s_axil_awaddr[CFG_ADDR_WIDTH+1:2];
    assign held_aw_word_addr = aw_fire ? aw_word_addr : awaddr_hold[CFG_ADDR_WIDTH+1:2];
    assign ar_word_addr = s_axil_araddr[CFG_ADDR_WIDTH+1:2];

    assign aw_fire = s_axil_awvalid && s_axil_awready;
    assign w_fire = s_axil_wvalid && s_axil_wready;
    assign have_aw = aw_hold_valid || aw_fire;
    assign have_w = w_hold_valid || w_fire;
    assign write_is_status_space = (held_aw_word_addr >= STATUS_WORD);
    assign read_is_status = (ar_word_addr == STATUS_WORD);
    assign read_is_version = (ar_word_addr == VERSION_WORD);
    assign read_is_param = (ar_word_addr == PARAM_WORD);
    assign write_full_strobe = ((w_fire ? s_axil_wstrb : wstrb_hold) == {(DATA_WIDTH/8){1'b1}});
    assign cfg_addr_mux = cfg_we_r ? cfg_addr_r :
                          ((!rvalid_r && s_axil_arvalid) ? ar_word_addr : cfg_addr_r);

    assign s_axil_awready = awready_r;
    assign s_axil_wready = wready_r;
    assign s_axil_bvalid = bvalid_r;
    assign s_axil_bresp = bresp_r;
    assign s_axil_arready = arready_r;
    assign s_axil_rvalid = rvalid_r;
    assign s_axil_rresp = rresp_r;
    assign s_axil_rdata = rdata_r;

    assign s_axis_z_tready = core_in_ready;
    assign m_axis_x_tdata = core_x_out;
    assign m_axis_x_tvalid = core_out_valid;
    assign m_axis_x_tlast = core_out_valid;
    assign m_axis_x_tuser[0] = core_err_singular;

    kalman_filter_matrix #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_WIDTH(FRAC_WIDTH),
        .N(N),
        .M(M),
        .CFG_ADDR_WIDTH(CFG_ADDR_WIDTH)
    ) u_core (
        .clk(aclk),
        .rst_n(aresetn),
        .in_valid(s_axis_z_tvalid),
        .in_ready(core_in_ready),
        .z_in(s_axis_z_tdata),
        .out_valid(core_out_valid),
        .out_ready(m_axis_x_tready),
        .x_out(core_x_out),
        .err_singular(core_err_singular),
        .busy(core_busy),
        .cfg_we(cfg_we_r),
        .cfg_addr(cfg_addr_mux),
        .cfg_wdata(cfg_wdata_r),
        .cfg_rdata(cfg_rdata)
    );

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            awready_r <= 1'b0;
            wready_r <= 1'b0;
            bvalid_r <= 1'b0;
            bresp_r <= AXI_RESP_OKAY;
            arready_r <= 1'b0;
            rvalid_r <= 1'b0;
            rresp_r <= AXI_RESP_OKAY;
            rdata_r <= {DATA_WIDTH{1'b0}};
            aw_hold_valid <= 1'b0;
            w_hold_valid <= 1'b0;
            awaddr_hold <= {AXIL_ADDR_WIDTH{1'b0}};
            wdata_hold <= {DATA_WIDTH{1'b0}};
            wstrb_hold <= {(DATA_WIDTH/8){1'b0}};
            cfg_we_r <= 1'b0;
            cfg_addr_r <= {CFG_ADDR_WIDTH{1'b0}};
            cfg_wdata_r <= {DATA_WIDTH{1'b0}};
        end else begin
            awready_r <= !aw_hold_valid && !bvalid_r;
            wready_r <= !w_hold_valid && !bvalid_r;
            arready_r <= 1'b0;
            cfg_we_r <= 1'b0;

            if (bvalid_r && s_axil_bready)
                bvalid_r <= 1'b0;

            if (rvalid_r && s_axil_rready)
                rvalid_r <= 1'b0;

            if (aw_fire) begin
                awaddr_hold <= s_axil_awaddr;
                aw_hold_valid <= 1'b1;
            end

            if (w_fire) begin
                wdata_hold <= s_axil_wdata;
                wstrb_hold <= s_axil_wstrb;
                w_hold_valid <= 1'b1;
            end

            if (!bvalid_r && have_aw && have_w) begin
                bvalid_r <= 1'b1;
                cfg_addr_r <= held_aw_word_addr;
                cfg_wdata_r <= w_fire ? s_axil_wdata : wdata_hold;
                aw_hold_valid <= 1'b0;
                w_hold_valid <= 1'b0;

                if (write_is_status_space || core_busy || !write_full_strobe) begin
                    bresp_r <= AXI_RESP_SLVERR;
                end else begin
                    cfg_we_r <= 1'b1;
                    bresp_r <= AXI_RESP_OKAY;
                end
            end

            if (!rvalid_r && s_axil_arvalid) begin
                arready_r <= 1'b1;
                rvalid_r <= 1'b1;
                rresp_r <= AXI_RESP_OKAY;
                cfg_addr_r <= ar_word_addr;

                if (read_is_status) begin
                    rdata_r <= {{(DATA_WIDTH-4){1'b0}},
                                core_err_singular,
                                core_out_valid,
                                core_in_ready,
                                core_busy};
                end else if (read_is_version) begin
                    rdata_r <= 32'h4B465001;
                end else if (read_is_param) begin
                    rdata_r <= {{(DATA_WIDTH-24){1'b0}},
                                N_U8,
                                M_U8,
                                DATA_WIDTH_U8};
                end else begin
                    rdata_r <= cfg_rdata;
                end
            end
        end
    end
endmodule
