`timescale 1ns/1ps

module tb_kalman_filter_axi;
    localparam DATA_WIDTH = 32;
    localparam FRAC_WIDTH = 16;
    localparam N = 4;
    localparam M = 2;
    localparam AXIL_ADDR_WIDTH = 12;

    localparam [AXIL_ADDR_WIDTH-1:0] REG_STATUS  = 12'h300;
    localparam [AXIL_ADDR_WIDTH-1:0] REG_VERSION = 12'h304;
    localparam [AXIL_ADDR_WIDTH-1:0] REG_PARAM   = 12'h308;
    localparam [AXIL_ADDR_WIDTH-1:0] REG_F0      = 12'h000;

    reg aclk;
    reg aresetn;

    reg [AXIL_ADDR_WIDTH-1:0] s_axil_awaddr;
    reg s_axil_awvalid;
    wire s_axil_awready;
    reg [DATA_WIDTH-1:0] s_axil_wdata;
    reg [(DATA_WIDTH/8)-1:0] s_axil_wstrb;
    reg s_axil_wvalid;
    wire s_axil_wready;
    wire [1:0] s_axil_bresp;
    wire s_axil_bvalid;
    reg s_axil_bready;

    reg [AXIL_ADDR_WIDTH-1:0] s_axil_araddr;
    reg s_axil_arvalid;
    wire s_axil_arready;
    wire [DATA_WIDTH-1:0] s_axil_rdata;
    wire [1:0] s_axil_rresp;
    wire s_axil_rvalid;
    reg s_axil_rready;

    reg [M*DATA_WIDTH-1:0] s_axis_z_tdata;
    reg s_axis_z_tvalid;
    wire s_axis_z_tready;
    reg s_axis_z_tlast;

    wire [N*DATA_WIDTH-1:0] m_axis_x_tdata;
    wire m_axis_x_tvalid;
    reg m_axis_x_tready;
    wire m_axis_x_tlast;
    wire [0:0] m_axis_x_tuser;

    reg [DATA_WIDTH-1:0] read_data;
    reg [1:0] read_resp;
    reg [1:0] write_resp;

    kalman_filter_axi #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_WIDTH(FRAC_WIDTH),
        .N(N),
        .M(M),
        .AXIL_ADDR_WIDTH(AXIL_ADDR_WIDTH)
    ) dut (
        .aclk(aclk),
        .aresetn(aresetn),
        .s_axil_awaddr(s_axil_awaddr),
        .s_axil_awvalid(s_axil_awvalid),
        .s_axil_awready(s_axil_awready),
        .s_axil_wdata(s_axil_wdata),
        .s_axil_wstrb(s_axil_wstrb),
        .s_axil_wvalid(s_axil_wvalid),
        .s_axil_wready(s_axil_wready),
        .s_axil_bresp(s_axil_bresp),
        .s_axil_bvalid(s_axil_bvalid),
        .s_axil_bready(s_axil_bready),
        .s_axil_araddr(s_axil_araddr),
        .s_axil_arvalid(s_axil_arvalid),
        .s_axil_arready(s_axil_arready),
        .s_axil_rdata(s_axil_rdata),
        .s_axil_rresp(s_axil_rresp),
        .s_axil_rvalid(s_axil_rvalid),
        .s_axil_rready(s_axil_rready),
        .s_axis_z_tdata(s_axis_z_tdata),
        .s_axis_z_tvalid(s_axis_z_tvalid),
        .s_axis_z_tready(s_axis_z_tready),
        .s_axis_z_tlast(s_axis_z_tlast),
        .m_axis_x_tdata(m_axis_x_tdata),
        .m_axis_x_tvalid(m_axis_x_tvalid),
        .m_axis_x_tready(m_axis_x_tready),
        .m_axis_x_tlast(m_axis_x_tlast),
        .m_axis_x_tuser(m_axis_x_tuser)
    );

    always #5 aclk = ~aclk;

    task axil_write;
        input [AXIL_ADDR_WIDTH-1:0] addr;
        input [DATA_WIDTH-1:0] data;
        begin
            @(posedge aclk);
            s_axil_awaddr <= addr;
            s_axil_wdata <= data;
            s_axil_wstrb <= {(DATA_WIDTH/8){1'b1}};
            s_axil_awvalid <= 1'b1;
            s_axil_wvalid <= 1'b1;
            s_axil_bready <= 1'b1;
            while (!(s_axil_awready && s_axil_wready))
                @(posedge aclk);
            @(posedge aclk);
            s_axil_awvalid <= 1'b0;
            s_axil_wvalid <= 1'b0;
            while (!s_axil_bvalid)
                @(posedge aclk);
            write_resp = s_axil_bresp;
            @(posedge aclk);
            s_axil_bready <= 1'b0;
        end
    endtask

    task axil_write_split;
        input [AXIL_ADDR_WIDTH-1:0] addr;
        input [DATA_WIDTH-1:0] data;
        begin
            @(posedge aclk);
            s_axil_awaddr <= addr;
            s_axil_awvalid <= 1'b1;
            while (!s_axil_awready)
                @(posedge aclk);
            @(posedge aclk);
            s_axil_awvalid <= 1'b0;

            repeat (3) @(posedge aclk);

            s_axil_wdata <= data;
            s_axil_wstrb <= {(DATA_WIDTH/8){1'b1}};
            s_axil_wvalid <= 1'b1;
            s_axil_bready <= 1'b1;
            while (!s_axil_wready)
                @(posedge aclk);
            @(posedge aclk);
            s_axil_wvalid <= 1'b0;
            while (!s_axil_bvalid)
                @(posedge aclk);
            write_resp = s_axil_bresp;
            @(posedge aclk);
            s_axil_bready <= 1'b0;
        end
    endtask


    task axil_read;
        input [AXIL_ADDR_WIDTH-1:0] addr;
        begin
            @(posedge aclk);
            s_axil_araddr <= addr;
            s_axil_arvalid <= 1'b1;
            s_axil_rready <= 1'b1;
            while (!s_axil_arready)
                @(posedge aclk);
            @(posedge aclk);
            s_axil_arvalid <= 1'b0;
            while (!s_axil_rvalid)
                @(posedge aclk);
            read_data = s_axil_rdata;
            read_resp = s_axil_rresp;
            @(posedge aclk);
            s_axil_rready <= 1'b0;
        end
    endtask

    task axis_send;
        input signed [DATA_WIDTH-1:0] z0;
        input signed [DATA_WIDTH-1:0] z1;
        begin
            @(posedge aclk);
            s_axis_z_tdata <= {z1, z0};
            s_axis_z_tvalid <= 1'b1;
            s_axis_z_tlast <= 1'b1;
            while (!s_axis_z_tready)
                @(posedge aclk);
            @(posedge aclk);
            s_axis_z_tvalid <= 1'b0;
            s_axis_z_tlast <= 1'b0;
            s_axis_z_tdata <= {M*DATA_WIDTH{1'b0}};
        end
    endtask

    initial begin
        aclk = 1'b0;
        aresetn = 1'b0;
        s_axil_awaddr = {AXIL_ADDR_WIDTH{1'b0}};
        s_axil_awvalid = 1'b0;
        s_axil_wdata = {DATA_WIDTH{1'b0}};
        s_axil_wstrb = {(DATA_WIDTH/8){1'b0}};
        s_axil_wvalid = 1'b0;
        s_axil_bready = 1'b0;
        s_axil_araddr = {AXIL_ADDR_WIDTH{1'b0}};
        s_axil_arvalid = 1'b0;
        s_axil_rready = 1'b0;
        s_axis_z_tdata = {M*DATA_WIDTH{1'b0}};
        s_axis_z_tvalid = 1'b0;
        s_axis_z_tlast = 1'b0;
        m_axis_x_tready = 1'b1;
        read_data = {DATA_WIDTH{1'b0}};
        read_resp = 2'b00;
        write_resp = 2'b00;

        repeat (5) @(posedge aclk);
        aresetn <= 1'b1;
        repeat (2) @(posedge aclk);

        axil_read(REG_VERSION);
        if (read_resp != 2'b00 || read_data != 32'h4B465001) begin
            $display("FAIL: AXI-Lite version read mismatch");
            $finish;
        end

        axil_read(REG_PARAM);
        if (read_resp != 2'b00 || read_data[23:0] != {8'd4, 8'd2, 8'd32}) begin
            $display("FAIL: AXI-Lite parameter read mismatch");
            $finish;
        end

        axil_write(REG_F0, 32'sd65536);
        if (write_resp != 2'b00) begin
            $display("FAIL: AXI-Lite idle write returned error");
            $finish;
        end

        axil_write_split(REG_F0, 32'sd65536);
        if (write_resp != 2'b00) begin
            $display("FAIL: split AXI-Lite write returned error");
            $finish;
        end

        axis_send(32'sd65536, 32'sd65536);
        axil_read(REG_STATUS);
        if (read_data[0] != 1'b1) begin
            $display("FAIL: status busy bit did not set after stream input");
            $finish;
        end

        axil_write(REG_F0, 32'sd65536);
        if (write_resp != 2'b10) begin
            $display("FAIL: busy AXI-Lite write did not return SLVERR");
            $finish;
        end

        while (!m_axis_x_tvalid)
            @(posedge aclk);
        if (!m_axis_x_tlast) begin
            $display("FAIL: AXI-Stream output missing TLAST");
            $finish;
        end
        if (^m_axis_x_tdata === 1'bx) begin
            $display("FAIL: AXI-Stream output contains X");
            $finish;
        end
        @(posedge aclk);

        axil_read(REG_STATUS);
        if (read_resp != 2'b00) begin
            $display("FAIL: status read returned error");
            $finish;
        end

        $display("PASS: kalman_filter_axi interface smoke test completed");
        $finish;
    end
endmodule
