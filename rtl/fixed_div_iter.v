`timescale 1ns/1ps

module fixed_div_iter #(
    parameter DATA_WIDTH = 32,
    parameter FRAC_WIDTH = 16
) (
    input clk,
    input rst_n,
    input start,
    input signed [DATA_WIDTH-1:0] numerator,
    input signed [DATA_WIDTH-1:0] denominator,
    output ready,
    output done,
    output signed [DATA_WIDTH-1:0] result,
    output div_zero,
    output overflow
);
    localparam [1:0] ST_IDLE = 2'd0;
    localparam [1:0] ST_RUN  = 2'd1;
    localparam [1:0] ST_DONE = 2'd2;

    reg [1:0] state;
    reg [6:0] bit_count;
    reg sign_q;
    reg [63:0] dividend_abs;
    reg [63:0] divisor_abs;
    reg [63:0] quotient_abs;
    reg [64:0] remainder;
    reg signed [DATA_WIDTH-1:0] result_r;
    reg div_zero_r;
    reg overflow_r;

    reg [64:0] rem_next;
    reg [63:0] quot_next;
    reg [63:0] max_abs_value;

    assign ready = (state == ST_IDLE);
    assign done = (state == ST_DONE);
    assign result = result_r;
    assign div_zero = div_zero_r;
    assign overflow = overflow_r;

    function [63:0] abs_extend;
        input signed [DATA_WIDTH-1:0] value;
        reg signed [63:0] ext;
        begin
            ext = {{(64-DATA_WIDTH){value[DATA_WIDTH-1]}}, value};
            abs_extend = ext[63] ? -ext : ext;
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            bit_count <= 7'd0;
            sign_q <= 1'b0;
            dividend_abs <= 64'd0;
            divisor_abs <= 64'd0;
            quotient_abs <= 64'd0;
            remainder <= 65'd0;
            result_r <= {DATA_WIDTH{1'b0}};
            div_zero_r <= 1'b0;
            overflow_r <= 1'b0;
        end else begin
            case (state)
                ST_IDLE: begin
                    div_zero_r <= 1'b0;
                    overflow_r <= 1'b0;
                    if (start) begin
                        sign_q <= numerator[DATA_WIDTH-1] ^ denominator[DATA_WIDTH-1];
                        dividend_abs <= abs_extend(numerator) << FRAC_WIDTH;
                        divisor_abs <= abs_extend(denominator);
                        quotient_abs <= 64'd0;
                        remainder <= 65'd0;
                        bit_count <= 7'd64;
                        if (denominator == {DATA_WIDTH{1'b0}}) begin
                            div_zero_r <= 1'b1;
                            result_r <= {DATA_WIDTH{1'b0}};
                            state <= ST_DONE;
                        end else begin
                            state <= ST_RUN;
                        end
                    end
                end

                ST_RUN: begin
                    rem_next = {remainder[63:0], dividend_abs[63]};
                    quot_next = {quotient_abs[62:0], 1'b0};
                    if (rem_next >= {1'b0, divisor_abs}) begin
                        rem_next = rem_next - {1'b0, divisor_abs};
                        quot_next[0] = 1'b1;
                    end

                    remainder <= rem_next;
                    quotient_abs <= quot_next;
                    dividend_abs <= {dividend_abs[62:0], 1'b0};
                    bit_count <= bit_count - 7'd1;

                    if (bit_count == 7'd1) begin
                        max_abs_value = sign_q ? (64'd1 << (DATA_WIDTH-1)) :
                                                 ((64'd1 << (DATA_WIDTH-1)) - 64'd1);
                        if (quot_next > max_abs_value) begin
                            result_r <= {1'b0, {DATA_WIDTH-1{1'b1}}};
                            overflow_r <= 1'b1;
                        end else begin
                            if (sign_q)
                                result_r <= -$signed(quot_next[DATA_WIDTH-1:0]);
                            else
                                result_r <= quot_next[DATA_WIDTH-1:0];
                        end
                        state <= ST_DONE;
                    end
                end

                ST_DONE: begin
                    state <= ST_IDLE;
                end

                default: begin
                    state <= ST_IDLE;
                end
            endcase
        end
    end
endmodule
