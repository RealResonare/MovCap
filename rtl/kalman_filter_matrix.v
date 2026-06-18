`timescale 1ns/1ps

module kalman_filter_matrix #(
    parameter DATA_WIDTH = 32,
    parameter FRAC_WIDTH = 16,
    parameter N = 4,
    parameter M = 2,
    parameter CFG_ADDR_WIDTH = 8
) (
    input clk,
    input rst_n,

    input in_valid,
    output in_ready,
    input signed [M*DATA_WIDTH-1:0] z_in,

    output out_valid,
    input out_ready,
    output signed [N*DATA_WIDTH-1:0] x_out,
    output err_singular,
    output busy,

    input cfg_we,
    input [CFG_ADDR_WIDTH-1:0] cfg_addr,
    input signed [DATA_WIDTH-1:0] cfg_wdata,
    output signed [DATA_WIDTH-1:0] cfg_rdata
);
    localparam [4:0] ST_IDLE       = 5'd0;
    localparam [4:0] ST_PREDICT_X  = 5'd1;
    localparam [4:0] ST_FP         = 5'd2;
    localparam [4:0] ST_PREDICT_P  = 5'd3;
    localparam [4:0] ST_INNOV      = 5'd4;
    localparam [4:0] ST_HP         = 5'd5;
    localparam [4:0] ST_S          = 5'd6;
    localparam [4:0] ST_DET0       = 5'd7;
    localparam [4:0] ST_DET1       = 5'd8;
    localparam [4:0] ST_DIV_START  = 5'd9;
    localparam [4:0] ST_DIV_WAIT   = 5'd10;
    localparam [4:0] ST_INV_WRITE  = 5'd11;
    localparam [4:0] ST_PHT        = 5'd12;
    localparam [4:0] ST_GAIN       = 5'd13;
    localparam [4:0] ST_UPDATE_X   = 5'd14;
    localparam [4:0] ST_KH         = 5'd15;
    localparam [4:0] ST_TMP_NP     = 5'd16;
    localparam [4:0] ST_TERM1      = 5'd17;
    localparam [4:0] ST_KR         = 5'd18;
    localparam [4:0] ST_UPDATE_P   = 5'd19;
    localparam [4:0] ST_OUTPUT     = 5'd20;

    localparam [CFG_ADDR_WIDTH-1:0] CFG_F_BASE = 8'h00;
    localparam [CFG_ADDR_WIDTH-1:0] CFG_H_BASE = 8'h20;
    localparam [CFG_ADDR_WIDTH-1:0] CFG_Q_BASE = 8'h40;
    localparam [CFG_ADDR_WIDTH-1:0] CFG_R_BASE = 8'h60;
    localparam [CFG_ADDR_WIDTH-1:0] CFG_X_BASE = 8'h80;
    localparam [CFG_ADDR_WIDTH-1:0] CFG_P_BASE = 8'hA0;

    localparam signed [DATA_WIDTH-1:0] ZERO_Q = {DATA_WIDTH{1'b0}};
    localparam signed [DATA_WIDTH-1:0] ONE_Q  = {{(DATA_WIDTH-FRAC_WIDTH-1){1'b0}}, 1'b1, {FRAC_WIDTH{1'b0}}};
    localparam signed [DATA_WIDTH-1:0] Q_DEFAULT = 32'sd66;
    localparam signed [DATA_WIDTH-1:0] R_DEFAULT = 32'sd16384;
    localparam signed [DATA_WIDTH-1:0] DET_EPS = 32'sd16;

    reg [4:0] state;
    reg out_valid_r;
    reg err_singular_r;

    reg signed [DATA_WIDTH-1:0] f_mat [0:N*N-1];
    reg signed [DATA_WIDTH-1:0] h_mat [0:M*N-1];
    reg signed [DATA_WIDTH-1:0] q_mat [0:N*N-1];
    reg signed [DATA_WIDTH-1:0] r_mat [0:M*M-1];
    reg signed [DATA_WIDTH-1:0] x_state [0:N-1];
    reg signed [DATA_WIDTH-1:0] p_state [0:N*N-1];

    reg signed [DATA_WIDTH-1:0] z_vec [0:M-1];
    reg signed [DATA_WIDTH-1:0] x_pred [0:N-1];
    reg signed [DATA_WIDTH-1:0] p_pred [0:N*N-1];
    reg signed [DATA_WIDTH-1:0] innovation [0:M-1];
    reg signed [DATA_WIDTH-1:0] s_mat [0:M*M-1];
    reg signed [DATA_WIDTH-1:0] s_inv [0:M*M-1];
    reg signed [DATA_WIDTH-1:0] gain [0:N*M-1];

    reg signed [DATA_WIDTH-1:0] fp_tmp [0:N*N-1];
    reg signed [DATA_WIDTH-1:0] hp_tmp [0:M*N-1];
    reg signed [DATA_WIDTH-1:0] ph_t_tmp [0:N*M-1];
    reg signed [DATA_WIDTH-1:0] kh_tmp [0:N*N-1];
    reg signed [DATA_WIDTH-1:0] i_kh_tmp [0:N*N-1];
    reg signed [DATA_WIDTH-1:0] tmp_np [0:N*N-1];
    reg signed [DATA_WIDTH-1:0] term1 [0:N*N-1];
    reg signed [DATA_WIDTH-1:0] kr_tmp [0:N*M-1];

    reg [7:0] row;
    reg [7:0] col;
    reg [7:0] inner;
    reg [7:0] inv_idx;
    reg signed [63:0] acc_q;
    reg signed [DATA_WIDTH-1:0] det_q;
    reg signed [DATA_WIDTH-1:0] det_inv;
    reg signed [DATA_WIDTH-1:0] det_prod0;
    reg signed [DATA_WIDTH-1:0] det_prod1;

    reg div_start;
    wire div_ready;
    wire div_done;
    wire div_div_zero;
    wire div_overflow;
    wire signed [DATA_WIDTH-1:0] div_result;

    reg signed [DATA_WIDTH-1:0] cfg_rdata_r;

    integer i;

    assign in_ready = (state == ST_IDLE) && !cfg_we;
    assign out_valid = out_valid_r;
    assign err_singular = err_singular_r;
    assign busy = (state != ST_IDLE);
    assign cfg_rdata = cfg_rdata_r;

    genvar xo;
    generate
        for (xo = 0; xo < N; xo = xo + 1) begin : pack_x_out
            assign x_out[(xo+1)*DATA_WIDTH-1:xo*DATA_WIDTH] = x_state[xo];
        end
    endgenerate

    fixed_div_iter #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_WIDTH(FRAC_WIDTH)
    ) u_det_div (
        .clk(clk),
        .rst_n(rst_n),
        .start(div_start),
        .numerator(ONE_Q),
        .denominator(det_q),
        .ready(div_ready),
        .done(div_done),
        .result(div_result),
        .div_zero(div_div_zero),
        .overflow(div_overflow)
    );

    function signed [63:0] to_s64;
        input signed [DATA_WIDTH-1:0] value;
        begin
            to_s64 = {{(64-DATA_WIDTH){value[DATA_WIDTH-1]}}, value};
        end
    endfunction

    function signed [DATA_WIDTH-1:0] sat64;
        input signed [63:0] value;
        reg signed [63:0] max_value;
        reg signed [63:0] min_value;
        begin
            max_value = (64'sd1 <<< (DATA_WIDTH-1)) - 64'sd1;
            min_value = -(64'sd1 <<< (DATA_WIDTH-1));
            if (value > max_value)
                sat64 = {1'b0, {DATA_WIDTH-1{1'b1}}};
            else if (value < min_value)
                sat64 = {1'b1, {DATA_WIDTH-1{1'b0}}};
            else
                sat64 = value[DATA_WIDTH-1:0];
        end
    endfunction

    function signed [DATA_WIDTH-1:0] add_sat;
        input signed [DATA_WIDTH-1:0] a;
        input signed [DATA_WIDTH-1:0] b;
        begin
            add_sat = sat64(to_s64(a) + to_s64(b));
        end
    endfunction

    function signed [DATA_WIDTH-1:0] sub_sat;
        input signed [DATA_WIDTH-1:0] a;
        input signed [DATA_WIDTH-1:0] b;
        begin
            sub_sat = sat64(to_s64(a) - to_s64(b));
        end
    endfunction

    function signed [DATA_WIDTH-1:0] neg_sat;
        input signed [DATA_WIDTH-1:0] a;
        begin
            neg_sat = sat64(-to_s64(a));
        end
    endfunction

    function signed [63:0] mul_q64;
        input signed [DATA_WIDTH-1:0] a;
        input signed [DATA_WIDTH-1:0] b;
        begin
            mul_q64 = (to_s64(a) * to_s64(b)) >>> FRAC_WIDTH;
        end
    endfunction

    function signed [DATA_WIDTH-1:0] mul_q;
        input signed [DATA_WIDTH-1:0] a;
        input signed [DATA_WIDTH-1:0] b;
        begin
            mul_q = sat64(mul_q64(a, b));
        end
    endfunction

    function signed [63:0] abs64;
        input signed [63:0] a;
        begin
            abs64 = a[63] ? -a : a;
        end
    endfunction

    function signed [DATA_WIDTH-1:0] mac_value;
        input signed [63:0] acc_in;
        input signed [DATA_WIDTH-1:0] a;
        input signed [DATA_WIDTH-1:0] b;
        begin
            mac_value = sat64(acc_in + mul_q64(a, b));
        end
    endfunction

    always @* begin
        cfg_rdata_r = ZERO_Q;
        if ((cfg_addr >= CFG_F_BASE) && (cfg_addr < CFG_F_BASE + N*N))
            cfg_rdata_r = f_mat[cfg_addr - CFG_F_BASE];
        else if ((cfg_addr >= CFG_H_BASE) && (cfg_addr < CFG_H_BASE + M*N))
            cfg_rdata_r = h_mat[cfg_addr - CFG_H_BASE];
        else if ((cfg_addr >= CFG_Q_BASE) && (cfg_addr < CFG_Q_BASE + N*N))
            cfg_rdata_r = q_mat[cfg_addr - CFG_Q_BASE];
        else if ((cfg_addr >= CFG_R_BASE) && (cfg_addr < CFG_R_BASE + M*M))
            cfg_rdata_r = r_mat[cfg_addr - CFG_R_BASE];
        else if ((cfg_addr >= CFG_X_BASE) && (cfg_addr < CFG_X_BASE + N))
            cfg_rdata_r = x_state[cfg_addr - CFG_X_BASE];
        else if ((cfg_addr >= CFG_P_BASE) && (cfg_addr < CFG_P_BASE + N*N))
            cfg_rdata_r = p_state[cfg_addr - CFG_P_BASE];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            out_valid_r <= 1'b0;
            err_singular_r <= 1'b0;
            row <= 8'd0;
            col <= 8'd0;
            inner <= 8'd0;
            inv_idx <= 8'd0;
            acc_q <= 64'sd0;
            det_q <= ZERO_Q;
            det_inv <= ZERO_Q;
            det_prod0 <= ZERO_Q;
            det_prod1 <= ZERO_Q;
            div_start <= 1'b0;

            for (i = 0; i < N*N; i = i + 1) begin
                f_mat[i] <= ZERO_Q;
                q_mat[i] <= ZERO_Q;
                p_state[i] <= ZERO_Q;
                p_pred[i] <= ZERO_Q;
                fp_tmp[i] <= ZERO_Q;
                kh_tmp[i] <= ZERO_Q;
                i_kh_tmp[i] <= ZERO_Q;
                tmp_np[i] <= ZERO_Q;
                term1[i] <= ZERO_Q;
            end
            for (i = 0; i < M*N; i = i + 1) begin
                h_mat[i] <= ZERO_Q;
                hp_tmp[i] <= ZERO_Q;
            end
            for (i = 0; i < M*M; i = i + 1) begin
                r_mat[i] <= ZERO_Q;
                s_mat[i] <= ZERO_Q;
                s_inv[i] <= ZERO_Q;
            end
            for (i = 0; i < N; i = i + 1) begin
                x_state[i] <= ZERO_Q;
                x_pred[i] <= ZERO_Q;
            end
            for (i = 0; i < M; i = i + 1) begin
                z_vec[i] <= ZERO_Q;
                innovation[i] <= ZERO_Q;
            end
            for (i = 0; i < N*M; i = i + 1) begin
                ph_t_tmp[i] <= ZERO_Q;
                gain[i] <= ZERO_Q;
                kr_tmp[i] <= ZERO_Q;
            end

            for (i = 0; i < N; i = i + 1) begin
                f_mat[i*N+i] <= ONE_Q;
                q_mat[i*N+i] <= Q_DEFAULT;
                p_state[i*N+i] <= ONE_Q;
            end
            if (N >= 2)
                f_mat[0*N+1] <= ONE_Q;
            if (N >= 4)
                f_mat[2*N+3] <= ONE_Q;
            if ((M >= 1) && (N >= 1))
                h_mat[0*N+0] <= ONE_Q;
            if ((M >= 2) && (N >= 3))
                h_mat[1*N+2] <= ONE_Q;
            for (i = 0; i < M; i = i + 1)
                r_mat[i*M+i] <= R_DEFAULT;
        end else begin
            div_start <= 1'b0;

            if ((state == ST_IDLE) && cfg_we) begin
                if ((cfg_addr >= CFG_F_BASE) && (cfg_addr < CFG_F_BASE + N*N))
                    f_mat[cfg_addr - CFG_F_BASE] <= cfg_wdata;
                else if ((cfg_addr >= CFG_H_BASE) && (cfg_addr < CFG_H_BASE + M*N))
                    h_mat[cfg_addr - CFG_H_BASE] <= cfg_wdata;
                else if ((cfg_addr >= CFG_Q_BASE) && (cfg_addr < CFG_Q_BASE + N*N))
                    q_mat[cfg_addr - CFG_Q_BASE] <= cfg_wdata;
                else if ((cfg_addr >= CFG_R_BASE) && (cfg_addr < CFG_R_BASE + M*M))
                    r_mat[cfg_addr - CFG_R_BASE] <= cfg_wdata;
                else if ((cfg_addr >= CFG_X_BASE) && (cfg_addr < CFG_X_BASE + N))
                    x_state[cfg_addr - CFG_X_BASE] <= cfg_wdata;
                else if ((cfg_addr >= CFG_P_BASE) && (cfg_addr < CFG_P_BASE + N*N))
                    p_state[cfg_addr - CFG_P_BASE] <= cfg_wdata;
            end

            case (state)
                ST_IDLE: begin
                    out_valid_r <= 1'b0;
                    err_singular_r <= 1'b0;
                    row <= 8'd0;
                    col <= 8'd0;
                    inner <= 8'd0;
                    acc_q <= 64'sd0;
                    if (in_valid && in_ready) begin
                        for (i = 0; i < M; i = i + 1)
                            z_vec[i] <= z_in[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH];
                        state <= ST_PREDICT_X;
                    end
                end

                ST_PREDICT_X: begin
                    if (inner == N-1) begin
                        x_pred[row] <= mac_value(acc_q, f_mat[row*N+inner], x_state[inner]);
                        acc_q <= 64'sd0;
                        inner <= 8'd0;
                        if (row == N-1) begin
                            row <= 8'd0;
                            col <= 8'd0;
                            state <= ST_FP;
                        end else begin
                            row <= row + 8'd1;
                        end
                    end else begin
                        acc_q <= acc_q + mul_q64(f_mat[row*N+inner], x_state[inner]);
                        inner <= inner + 8'd1;
                    end
                end

                ST_FP: begin
                    if (inner == N-1) begin
                        fp_tmp[row*N+col] <= mac_value(acc_q, f_mat[row*N+inner], p_state[inner*N+col]);
                        acc_q <= 64'sd0;
                        inner <= 8'd0;
                        if (col == N-1) begin
                            col <= 8'd0;
                            if (row == N-1) begin
                                row <= 8'd0;
                                state <= ST_PREDICT_P;
                            end else begin
                                row <= row + 8'd1;
                            end
                        end else begin
                            col <= col + 8'd1;
                        end
                    end else begin
                        acc_q <= acc_q + mul_q64(f_mat[row*N+inner], p_state[inner*N+col]);
                        inner <= inner + 8'd1;
                    end
                end

                ST_PREDICT_P: begin
                    if (inner == N-1) begin
                        p_pred[row*N+col] <= add_sat(mac_value(acc_q, fp_tmp[row*N+inner], f_mat[col*N+inner]),
                                                     q_mat[row*N+col]);
                        acc_q <= 64'sd0;
                        inner <= 8'd0;
                        if (col == N-1) begin
                            col <= 8'd0;
                            if (row == N-1) begin
                                row <= 8'd0;
                                state <= ST_INNOV;
                            end else begin
                                row <= row + 8'd1;
                            end
                        end else begin
                            col <= col + 8'd1;
                        end
                    end else begin
                        acc_q <= acc_q + mul_q64(fp_tmp[row*N+inner], f_mat[col*N+inner]);
                        inner <= inner + 8'd1;
                    end
                end

                ST_INNOV: begin
                    if (inner == N-1) begin
                        innovation[row] <= sub_sat(z_vec[row],
                                                   mac_value(acc_q, h_mat[row*N+inner], x_pred[inner]));
                        acc_q <= 64'sd0;
                        inner <= 8'd0;
                        if (row == M-1) begin
                            row <= 8'd0;
                            col <= 8'd0;
                            state <= ST_HP;
                        end else begin
                            row <= row + 8'd1;
                        end
                    end else begin
                        acc_q <= acc_q + mul_q64(h_mat[row*N+inner], x_pred[inner]);
                        inner <= inner + 8'd1;
                    end
                end

                ST_HP: begin
                    if (inner == N-1) begin
                        hp_tmp[row*N+col] <= mac_value(acc_q, h_mat[row*N+inner], p_pred[inner*N+col]);
                        acc_q <= 64'sd0;
                        inner <= 8'd0;
                        if (col == N-1) begin
                            col <= 8'd0;
                            if (row == M-1) begin
                                row <= 8'd0;
                                state <= ST_S;
                            end else begin
                                row <= row + 8'd1;
                            end
                        end else begin
                            col <= col + 8'd1;
                        end
                    end else begin
                        acc_q <= acc_q + mul_q64(h_mat[row*N+inner], p_pred[inner*N+col]);
                        inner <= inner + 8'd1;
                    end
                end

                ST_S: begin
                    if (inner == N-1) begin
                        s_mat[row*M+col] <= add_sat(mac_value(acc_q, hp_tmp[row*N+inner], h_mat[col*N+inner]),
                                                    r_mat[row*M+col]);
                        acc_q <= 64'sd0;
                        inner <= 8'd0;
                        if (col == M-1) begin
                            col <= 8'd0;
                            if (row == M-1) begin
                                row <= 8'd0;
                                state <= ST_DET0;
                            end else begin
                                row <= row + 8'd1;
                            end
                        end else begin
                            col <= col + 8'd1;
                        end
                    end else begin
                        acc_q <= acc_q + mul_q64(hp_tmp[row*N+inner], h_mat[col*N+inner]);
                        inner <= inner + 8'd1;
                    end
                end

                ST_DET0: begin
                    if (M != 2) begin
                        err_singular_r <= 1'b1;
                        out_valid_r <= 1'b1;
                        state <= ST_OUTPUT;
                    end else begin
                        det_prod0 <= mul_q(s_mat[0], s_mat[3]);
                        state <= ST_DET1;
                    end
                end

                ST_DET1: begin
                    det_prod1 <= mul_q(s_mat[1], s_mat[2]);
                    det_q <= sub_sat(det_prod0, mul_q(s_mat[1], s_mat[2]));
                    if (abs64(to_s64(sub_sat(det_prod0, mul_q(s_mat[1], s_mat[2])))) <= to_s64(DET_EPS)) begin
                        err_singular_r <= 1'b1;
                        out_valid_r <= 1'b1;
                        state <= ST_OUTPUT;
                    end else begin
                        state <= ST_DIV_START;
                    end
                end

                ST_DIV_START: begin
                    if (div_ready) begin
                        div_start <= 1'b1;
                        state <= ST_DIV_WAIT;
                    end
                end

                ST_DIV_WAIT: begin
                    if (div_done) begin
                        det_inv <= div_result;
                        if (div_div_zero || div_overflow) begin
                            err_singular_r <= 1'b1;
                            out_valid_r <= 1'b1;
                            state <= ST_OUTPUT;
                        end else begin
                            inv_idx <= 8'd0;
                            state <= ST_INV_WRITE;
                        end
                    end
                end

                ST_INV_WRITE: begin
                    if (inv_idx == 8'd0)
                        s_inv[0] <= mul_q(s_mat[3], det_inv);
                    else if (inv_idx == 8'd1)
                        s_inv[1] <= mul_q(neg_sat(s_mat[1]), det_inv);
                    else if (inv_idx == 8'd2)
                        s_inv[2] <= mul_q(neg_sat(s_mat[2]), det_inv);
                    else
                        s_inv[3] <= mul_q(s_mat[0], det_inv);

                    if (inv_idx == 8'd3) begin
                        inv_idx <= 8'd0;
                        row <= 8'd0;
                        col <= 8'd0;
                        inner <= 8'd0;
                        acc_q <= 64'sd0;
                        state <= ST_PHT;
                    end else begin
                        inv_idx <= inv_idx + 8'd1;
                    end
                end

                ST_PHT: begin
                    if (inner == N-1) begin
                        ph_t_tmp[row*M+col] <= mac_value(acc_q, p_pred[row*N+inner], h_mat[col*N+inner]);
                        acc_q <= 64'sd0;
                        inner <= 8'd0;
                        if (col == M-1) begin
                            col <= 8'd0;
                            if (row == N-1) begin
                                row <= 8'd0;
                                state <= ST_GAIN;
                            end else begin
                                row <= row + 8'd1;
                            end
                        end else begin
                            col <= col + 8'd1;
                        end
                    end else begin
                        acc_q <= acc_q + mul_q64(p_pred[row*N+inner], h_mat[col*N+inner]);
                        inner <= inner + 8'd1;
                    end
                end

                ST_GAIN: begin
                    if (inner == M-1) begin
                        gain[row*M+col] <= mac_value(acc_q, ph_t_tmp[row*M+inner], s_inv[inner*M+col]);
                        acc_q <= 64'sd0;
                        inner <= 8'd0;
                        if (col == M-1) begin
                            col <= 8'd0;
                            if (row == N-1) begin
                                row <= 8'd0;
                                state <= ST_UPDATE_X;
                            end else begin
                                row <= row + 8'd1;
                            end
                        end else begin
                            col <= col + 8'd1;
                        end
                    end else begin
                        acc_q <= acc_q + mul_q64(ph_t_tmp[row*M+inner], s_inv[inner*M+col]);
                        inner <= inner + 8'd1;
                    end
                end

                ST_UPDATE_X: begin
                    if (inner == M-1) begin
                        x_state[row] <= add_sat(x_pred[row], mac_value(acc_q, gain[row*M+inner], innovation[inner]));
                        acc_q <= 64'sd0;
                        inner <= 8'd0;
                        if (row == N-1) begin
                            row <= 8'd0;
                            col <= 8'd0;
                            state <= ST_KH;
                        end else begin
                            row <= row + 8'd1;
                        end
                    end else begin
                        acc_q <= acc_q + mul_q64(gain[row*M+inner], innovation[inner]);
                        inner <= inner + 8'd1;
                    end
                end

                ST_KH: begin
                    if (inner == M-1) begin
                        kh_tmp[row*N+col] <= mac_value(acc_q, gain[row*M+inner], h_mat[inner*N+col]);
                        if (row == col)
                            i_kh_tmp[row*N+col] <= sub_sat(ONE_Q, mac_value(acc_q, gain[row*M+inner], h_mat[inner*N+col]));
                        else
                            i_kh_tmp[row*N+col] <= neg_sat(mac_value(acc_q, gain[row*M+inner], h_mat[inner*N+col]));
                        acc_q <= 64'sd0;
                        inner <= 8'd0;
                        if (col == N-1) begin
                            col <= 8'd0;
                            if (row == N-1) begin
                                row <= 8'd0;
                                state <= ST_TMP_NP;
                            end else begin
                                row <= row + 8'd1;
                            end
                        end else begin
                            col <= col + 8'd1;
                        end
                    end else begin
                        acc_q <= acc_q + mul_q64(gain[row*M+inner], h_mat[inner*N+col]);
                        inner <= inner + 8'd1;
                    end
                end

                ST_TMP_NP: begin
                    if (inner == N-1) begin
                        tmp_np[row*N+col] <= mac_value(acc_q, i_kh_tmp[row*N+inner], p_pred[inner*N+col]);
                        acc_q <= 64'sd0;
                        inner <= 8'd0;
                        if (col == N-1) begin
                            col <= 8'd0;
                            if (row == N-1) begin
                                row <= 8'd0;
                                state <= ST_TERM1;
                            end else begin
                                row <= row + 8'd1;
                            end
                        end else begin
                            col <= col + 8'd1;
                        end
                    end else begin
                        acc_q <= acc_q + mul_q64(i_kh_tmp[row*N+inner], p_pred[inner*N+col]);
                        inner <= inner + 8'd1;
                    end
                end

                ST_TERM1: begin
                    if (inner == N-1) begin
                        term1[row*N+col] <= mac_value(acc_q, tmp_np[row*N+inner], i_kh_tmp[col*N+inner]);
                        acc_q <= 64'sd0;
                        inner <= 8'd0;
                        if (col == N-1) begin
                            col <= 8'd0;
                            if (row == N-1) begin
                                row <= 8'd0;
                                state <= ST_KR;
                            end else begin
                                row <= row + 8'd1;
                            end
                        end else begin
                            col <= col + 8'd1;
                        end
                    end else begin
                        acc_q <= acc_q + mul_q64(tmp_np[row*N+inner], i_kh_tmp[col*N+inner]);
                        inner <= inner + 8'd1;
                    end
                end

                ST_KR: begin
                    if (inner == M-1) begin
                        kr_tmp[row*M+col] <= mac_value(acc_q, gain[row*M+inner], r_mat[inner*M+col]);
                        acc_q <= 64'sd0;
                        inner <= 8'd0;
                        if (col == M-1) begin
                            col <= 8'd0;
                            if (row == N-1) begin
                                row <= 8'd0;
                                state <= ST_UPDATE_P;
                            end else begin
                                row <= row + 8'd1;
                            end
                        end else begin
                            col <= col + 8'd1;
                        end
                    end else begin
                        acc_q <= acc_q + mul_q64(gain[row*M+inner], r_mat[inner*M+col]);
                        inner <= inner + 8'd1;
                    end
                end

                ST_UPDATE_P: begin
                    if (inner == M-1) begin
                        p_state[row*N+col] <= add_sat(term1[row*N+col],
                                                       mac_value(acc_q, kr_tmp[row*M+inner], gain[col*M+inner]));
                        acc_q <= 64'sd0;
                        inner <= 8'd0;
                        if (col == N-1) begin
                            col <= 8'd0;
                            if (row == N-1) begin
                                row <= 8'd0;
                                out_valid_r <= 1'b1;
                                state <= ST_OUTPUT;
                            end else begin
                                row <= row + 8'd1;
                            end
                        end else begin
                            col <= col + 8'd1;
                        end
                    end else begin
                        acc_q <= acc_q + mul_q64(kr_tmp[row*M+inner], gain[col*M+inner]);
                        inner <= inner + 8'd1;
                    end
                end

                ST_OUTPUT: begin
                    if (out_ready) begin
                        out_valid_r <= 1'b0;
                        state <= ST_IDLE;
                    end
                end

                default: begin
                    state <= ST_IDLE;
                    out_valid_r <= 1'b0;
                end
            endcase
        end
    end
endmodule
