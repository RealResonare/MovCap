`timescale 1ns/1ps

module fixed_mul #(
    parameter DATA_WIDTH = 32,
    parameter FRAC_WIDTH = 16
) (
    input  signed [DATA_WIDTH-1:0] a,
    input  signed [DATA_WIDTH-1:0] b,
    output signed [DATA_WIDTH-1:0] result,
    output overflow
);
    reg signed [63:0] product;
    reg signed [63:0] shifted;
    reg signed [63:0] max_value;
    reg signed [63:0] min_value;
    reg signed [DATA_WIDTH-1:0] result_r;
    reg overflow_r;

    always @* begin
        product = {{(64-DATA_WIDTH){a[DATA_WIDTH-1]}}, a} *
                  {{(64-DATA_WIDTH){b[DATA_WIDTH-1]}}, b};
        shifted = product >>> FRAC_WIDTH;

        max_value = (64'sd1 <<< (DATA_WIDTH-1)) - 64'sd1;
        min_value = -(64'sd1 <<< (DATA_WIDTH-1));

        if (shifted > max_value) begin
            result_r = {1'b0, {DATA_WIDTH-1{1'b1}}};
            overflow_r = 1'b1;
        end else if (shifted < min_value) begin
            result_r = {1'b1, {DATA_WIDTH-1{1'b0}}};
            overflow_r = 1'b1;
        end else begin
            result_r = shifted[DATA_WIDTH-1:0];
            overflow_r = 1'b0;
        end
    end

    assign result = result_r;
    assign overflow = overflow_r;
endmodule
