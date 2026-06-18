`timescale 1ns/1ps

module tb_kalman_filter_matrix;
    localparam DATA_WIDTH = 32;
    localparam FRAC_WIDTH = 16;
    localparam N = 4;
    localparam M = 2;

    localparam signed [DATA_WIDTH-1:0] Q_ZERO = 32'sd0;

    reg clk;
    reg rst_n;
    reg in_valid;
    wire in_ready;
    reg signed [M*DATA_WIDTH-1:0] z_in;
    wire out_valid;
    reg out_ready;
    wire signed [N*DATA_WIDTH-1:0] x_out;
    wire err_singular;
    wire busy;
    reg cfg_we;
    reg [7:0] cfg_addr;
    reg signed [DATA_WIDTH-1:0] cfg_wdata;
    wire signed [DATA_WIDTH-1:0] cfg_rdata;

    integer output_count;
    integer cfg_loop;
    reg signed [N*DATA_WIDTH-1:0] hold_x;

    kalman_filter_matrix #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_WIDTH(FRAC_WIDTH),
        .N(N),
        .M(M)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .z_in(z_in),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .x_out(x_out),
        .err_singular(err_singular),
        .busy(busy),
        .cfg_we(cfg_we),
        .cfg_addr(cfg_addr),
        .cfg_wdata(cfg_wdata),
        .cfg_rdata(cfg_rdata)
    );

    always #5 clk = ~clk;

    task cfg_write;
        input [7:0] addr;
        input signed [DATA_WIDTH-1:0] data;
        begin
            @(posedge clk);
            cfg_addr <= addr;
            cfg_wdata <= data;
            cfg_we <= 1'b1;
            @(posedge clk);
            cfg_we <= 1'b0;
            cfg_addr <= 8'h00;
            cfg_wdata <= Q_ZERO;
        end
    endtask

    task send_measurement;
        input signed [DATA_WIDTH-1:0] z0;
        input signed [DATA_WIDTH-1:0] z1;
        begin
            while (!in_ready)
                @(posedge clk);
            @(posedge clk);
            z_in <= {z1, z0};
            in_valid <= 1'b1;
            @(posedge clk);
            in_valid <= 1'b0;
            z_in <= {M*DATA_WIDTH{1'b0}};
        end
    endtask

    task wait_for_output;
        begin
            while (!out_valid)
                @(posedge clk);
            if (^x_out === 1'bx) begin
                $display("FAIL: x_out contains X");
                $finish;
            end
            @(posedge clk);
            output_count = output_count + 1;
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        in_valid = 1'b0;
        z_in = {M*DATA_WIDTH{1'b0}};
        out_ready = 1'b1;
        cfg_we = 1'b0;
        cfg_addr = 8'h00;
        cfg_wdata = Q_ZERO;
        output_count = 0;
        hold_x = {N*DATA_WIDTH{1'b0}};

        repeat (5) @(posedge clk);
        rst_n <= 1'b1;
        repeat (2) @(posedge clk);

        send_measurement(32'sd65536, 32'sd65536);
        wait_for_output();
        if (err_singular) begin
            $display("FAIL: normal update flagged singular");
            $finish;
        end

        send_measurement(32'sd131072, 32'sd98304);
        wait_for_output();
        if (err_singular) begin
            $display("FAIL: second normal update flagged singular");
            $finish;
        end

        out_ready <= 1'b0;
        send_measurement(32'sd196608, 32'sd131072);
        while (!out_valid)
            @(posedge clk);
        hold_x = x_out;
        repeat (4) begin
            @(posedge clk);
            if (x_out !== hold_x) begin
                $display("FAIL: x_out changed while out_ready was low");
                $finish;
            end
        end
        out_ready <= 1'b1;
        @(posedge clk);
        output_count = output_count + 1;

        for (cfg_loop = 8'h20; cfg_loop < 8'h28; cfg_loop = cfg_loop + 1)
            cfg_write(cfg_loop[7:0], Q_ZERO);
        for (cfg_loop = 8'h60; cfg_loop < 8'h64; cfg_loop = cfg_loop + 1)
            cfg_write(cfg_loop[7:0], Q_ZERO);

        send_measurement(32'sd262144, 32'sd196608);
        wait_for_output();
        if (!err_singular) begin
            $display("FAIL: singular S was not detected");
            $finish;
        end

        if (output_count < 4) begin
            $display("FAIL: missing outputs");
            $finish;
        end

        $display("PASS: kalman_filter_matrix smoke test completed");
        $finish;
    end
endmodule
