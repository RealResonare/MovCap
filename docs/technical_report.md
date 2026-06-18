# 基于 FPGA 的卡尔曼滤波 Verilog IP 技术说明

## 1. 项目概述

本项目实现了一套基于 FPGA 的卡尔曼滤波 Verilog IP 核，用于对二维位置测量数据进行状态估计和噪声抑制。设计采用 Verilog-2001 编写，默认状态维度为 `N=4`，测量维度为 `M=2`，定点数格式为有符号 `Q16.16`。工程重点不是将 CPU/GPU 中的矩阵运算过程直接复制到硬件中，而是将卡尔曼滤波算法映射为适合 FPGA 综合的寄存器、计数器、乘加数据通路和多周期除法单元。

默认模型为二维匀速运动模型：

- 状态向量：`x = [px, vx, py, vy]^T`
- 测量向量：`z = [px_meas, py_meas]^T`
- 状态转移矩阵：

```text
F = [ 1  1  0  0
      0  1  0  0
      0  0  1  1
      0  0  0  1 ]
```

- 观测矩阵：

```text
H = [ 1  0  0  0
      0  0  1  0 ]
```

工程主要文件如下：

| 文件 | 说明 |
| --- | --- |
| `rtl/kalman_filter_matrix.v` | 卡尔曼滤波顶层模块，包含配置寄存器、状态机、矩阵计算调度、输出握手 |
| `rtl/kalman_filter_axi.v` | AXI-Lite 配置接口和 AXI-Stream 数据接口封装 |
| `rtl/fixed_mul.v` | 定点乘法饱和模块，可独立复用 |
| `rtl/fixed_div_iter.v` | 多周期定点除法模块，用于 `2x2` 矩阵求逆中的行列式倒数 |
| `tb/tb_kalman_filter_matrix.v` | 基础 testbench，覆盖正常更新、输出背压、奇异矩阵检测 |
| `tb/tb_kalman_filter_axi.v` | AXI-Lite/AXI-Stream 接口 testbench，覆盖总线读写和流式输入输出 |
| `Makefile` | 仿真入口，默认使用 Icarus Verilog |
| `README.md` | 工程使用说明和地址映射摘要 |

## 2. 设计目标

本设计的目标包括：

1. 实现可综合的卡尔曼滤波 IP 核，能够在 FPGA 中完成预测、更新和状态输出。
2. 使用固定点数代替浮点数，降低资源消耗，便于映射到 FPGA DSP 与逻辑资源。
3. 采用 `valid/ready` 握手机制，并在外层封装为 AXI-Stream，方便与采样模块、DMA 或后级处理模块连接。
4. 提供寄存器式配置口，并在外层封装为 AXI-Lite，使 `F`、`H`、`Q`、`R`、状态向量 `x` 和协方差矩阵 `P` 可由处理器或总线主设备配置。
5. 采用逐项 MAC 和多周期除法的数据通路，避免在一个时钟周期内展开整块矩阵计算，满足 FPGA 流水/迭代式运算结构要求。
6. 采用“计算核 + 总线封装”的分层结构，后续可快速扩展接口、状态寄存器或并行 MAC 数据通路。

## 3. 算法原理

标准离散卡尔曼滤波由预测和更新两部分组成。

预测阶段：

```text
x_pred = F * x
P_pred = F * P * F^T + Q
```

更新阶段：

```text
y = z - H * x_pred
S = H * P_pred * H^T + R
K = P_pred * H^T * S^-1
x = x_pred + K * y
P = (I - K * H) * P_pred * (I - K * H)^T + K * R * K^T
```

其中：

- `x` 为状态估计向量。
- `P` 为状态估计协方差矩阵。
- `F` 为状态转移矩阵。
- `H` 为观测矩阵。
- `Q` 为过程噪声协方差矩阵。
- `R` 为测量噪声协方差矩阵。
- `K` 为卡尔曼增益。
- `S` 为测量残差协方差矩阵。

协方差更新采用 Joseph 形式：

```text
P = (I - K*H) * P_pred * (I-KH)^T + K*R*K^T
```

该形式相比简化形式 `P = (I - K*H) * P_pred` 数值稳定性更好，更适合固定点实现。

## 4. 数据格式与数值设计

### 4.1 定点格式

工程默认使用有符号 `Q16.16` 格式：

- 总位宽：32 bit
- 符号位：1 bit
- 整数部分：15 bit
- 小数部分：16 bit
- 数值 `1.0` 表示为 `32'sd65536`

采用固定点格式的原因：

1. FPGA 中固定点乘加可高效映射到 DSP 资源。
2. 相比浮点单元，固定点逻辑面积更小、时序更容易收敛。
3. 对常见传感器滤波和位置估计场景，`Q16.16` 能提供较好的动态范围和精度平衡。

### 4.2 乘法与累加

定点乘法遵循：

```text
Q16.16 * Q16.16 = Q32.32
Q32.32 >>> 16 = Q16.16
```

顶层模块中的 MAC 过程使用 64 位中间累加寄存器 `acc_q`，每拍累加一个乘积项：

```verilog
acc_q <= acc_q + mul_q64(a, b);
```

点积末项完成后，通过饱和函数限制到 `DATA_WIDTH` 位输出，避免固定点溢出导致符号翻转。

### 4.3 除法

卡尔曼增益计算需要求 `S^-1`。默认 `M=2`，因此 `S` 为 `2x2` 矩阵：

```text
S = [ s00  s01
      s10  s11 ]
```

其逆矩阵为：

```text
S^-1 = 1/det(S) * [  s11  -s01
                    -s10   s00 ]
det(S) = s00*s11 - s01*s10
```

工程使用 `fixed_div_iter.v` 计算 `1/det(S)`，该模块为多周期恢复除法器，不使用组合 `/` 作为顶层关键路径。这样可以避免综合器生成面积大、时序难收敛的组合除法网络。

## 5. FPGA 硬件结构

### 5.1 顶层模块接口

顶层模块为 `kalman_filter_matrix`，主要参数如下：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `DATA_WIDTH` | 32 | 定点数据总位宽 |
| `FRAC_WIDTH` | 16 | 小数位宽 |
| `N` | 4 | 状态向量维度 |
| `M` | 2 | 测量向量维度 |
| `CFG_ADDR_WIDTH` | 8 | 配置地址位宽 |

主要端口如下：

| 端口 | 方向 | 说明 |
| --- | --- | --- |
| `clk` | input | 系统时钟 |
| `rst_n` | input | 低有效复位 |
| `in_valid` | input | 输入测量数据有效 |
| `in_ready` | output | 模块可接收输入 |
| `z_in` | input | 打包后的测量向量 |
| `out_valid` | output | 输出状态有效 |
| `out_ready` | input | 下游可接收输出 |
| `x_out` | output | 打包后的状态估计向量 |
| `err_singular` | output | `S` 矩阵奇异或除法异常标志 |
| `busy` | output | 模块正在处理当前测量帧 |
| `cfg_we` | input | 配置写使能 |
| `cfg_addr` | input | 配置地址 |
| `cfg_wdata` | input | 配置写数据 |
| `cfg_rdata` | output | 配置读数据 |

### 5.2 配置地址映射

所有矩阵均采用行优先方式存储，数据格式均为 `Q16.16`。

| 地址范围 | 内容 | 数量 |
| --- | --- | --- |
| `0x00..0x0F` | `F` 矩阵 | 16 words |
| `0x20..0x27` | `H` 矩阵 | 8 words |
| `0x40..0x4F` | `Q` 矩阵 | 16 words |
| `0x60..0x63` | `R` 矩阵 | 4 words |
| `0x80..0x83` | 状态向量 `x` | 4 words |
| `0xA0..0xAF` | 协方差矩阵 `P` | 16 words |

配置写入仅在 `state == ST_IDLE` 时生效，避免处理过程中矩阵被修改导致结果不一致。

### 5.3 AXI-Lite 配置封装

新增 `kalman_filter_axi.v` 作为系统集成顶层。该模块将计算核原有的 `cfg_we/cfg_addr/cfg_wdata/cfg_rdata` 简化配置口封装为 AXI-Lite 从接口。AXI-Lite 地址采用字节地址，内部通过 `addr[CFG_ADDR_WIDTH+1:2]` 转换为计算核使用的字地址。

AXI-Lite 字节地址映射如下：

| 地址范围 | 内容 |
| --- | --- |
| `0x000..0x03C` | `F` 矩阵 |
| `0x080..0x09C` | `H` 矩阵 |
| `0x100..0x13C` | `Q` 矩阵 |
| `0x180..0x18C` | `R` 矩阵 |
| `0x200..0x20C` | 状态向量 `x` |
| `0x280..0x2BC` | 协方差矩阵 `P` |
| `0x300` | 状态寄存器 `{err_singular, out_valid, in_ready, busy}` |
| `0x304` | 版本寄存器 `0x4B465001` |
| `0x308` | 参数寄存器 `{N, M, DATA_WIDTH}` |

AXI-Lite 写通道支持 AW 和 W 不同拍到达，封装内部使用保持寄存器缓存地址和数据。配置矩阵时要求计算核处于空闲状态；若在 `busy=1` 时写入配置空间，AXI-Lite 返回 `SLVERR`，避免运行中修改滤波参数。

### 5.4 AXI-Stream 数据封装

AXI-Stream 输入端口 `s_axis_z_tdata` 对应一个完整测量向量，默认宽度为 `M*DATA_WIDTH=64` bit，打包顺序与计算核 `z_in` 一致。输入握手：

```text
s_axis_z_tvalid && s_axis_z_tready
```

成立时，模块接收一帧测量数据。当前实现每次只接收一个完整测量向量，因此 `s_axis_z_tlast` 作为帧结束标志保留，计算核默认不依赖其数值。

AXI-Stream 输出端口 `m_axis_x_tdata` 对应一个完整状态向量，默认宽度为 `N*DATA_WIDTH=128` bit。输出握手：

```text
m_axis_x_tvalid && m_axis_x_tready
```

成立时，下游接收一帧滤波结果。`m_axis_x_tlast` 在输出有效时置位，`m_axis_x_tuser[0]` 携带 `err_singular` 异常标志，方便下游模块或 DMA 软件层识别异常样本。

## 6. 流水/迭代式运算设计

### 6.1 非 CPU/GPU 式实现说明

CPU/GPU 中常见做法是通过软件循环遍历矩阵元素，并依靠指令执行单元或 SIMD/SIMT 单元完成矩阵运算。如果将这种写法直接翻译到 Verilog 中，例如在一个时钟状态内写多层嵌套 `for` 完成整块矩阵乘法，综合器通常会将其展开为大量并行组合乘加逻辑。这会导致以下问题：

- 组合路径过长，时钟频率难以提高。
- 乘法器数量激增，DSP 资源消耗大。
- 难以控制吞吐率与面积之间的平衡。
- 设计结构不清晰，不利于后续做流水线或资源复用。

当前版本已避免这种结构。矩阵运算由 `row`、`col`、`inner` 三个计数器调度，每拍推进一个点积项，形成固定数据通路：

```text
读矩阵元素 -> 定点乘法 -> 累加寄存器 -> 点积完成 -> 写回结果矩阵
```

该结构更符合 FPGA 设计方式：

- 使用寄存器保存中间状态。
- 使用有限状态机控制计算阶段。
- 使用 MAC 数据通路逐项完成矩阵运算。
- 使用多周期除法降低关键路径压力。

### 6.2 状态机阶段

顶层状态机主要阶段如下：

| 状态 | 功能 |
| --- | --- |
| `ST_IDLE` | 等待输入测量或配置写入 |
| `ST_PREDICT_X` | 计算 `x_pred = F*x` |
| `ST_FP` | 计算中间矩阵 `F*P` |
| `ST_PREDICT_P` | 计算 `P_pred = F*P*F^T + Q` |
| `ST_INNOV` | 计算残差 `y = z - H*x_pred` |
| `ST_HP` | 计算中间矩阵 `H*P_pred` |
| `ST_S` | 计算 `S = H*P_pred*H^T + R` |
| `ST_DET0/ST_DET1` | 计算 `2x2` 矩阵行列式 |
| `ST_DIV_START/ST_DIV_WAIT` | 调用迭代除法器计算 `1/det(S)` |
| `ST_INV_WRITE` | 写入 `S^-1` 四个元素 |
| `ST_PHT` | 计算 `P_pred*H^T` |
| `ST_GAIN` | 计算卡尔曼增益 `K` |
| `ST_UPDATE_X` | 更新状态向量 |
| `ST_KH` | 计算 `K*H` 和 `I-KH` |
| `ST_TMP_NP` | 计算 `(I-KH)*P_pred` |
| `ST_TERM1` | 计算 `(I-KH)*P_pred*(I-KH)^T` |
| `ST_KR` | 计算 `K*R` |
| `ST_UPDATE_P` | 完成 Joseph 协方差更新 |
| `ST_OUTPUT` | 输出结果并等待下游接收 |

每个矩阵点积都由 `inner` 计数器逐项推进。以 `ST_PREDICT_X` 为例，状态机不会在一拍内完成全部 `F*x`，而是每拍完成一个乘加项，点积结束后写入一个 `x_pred[row]`，随后推进到下一行。

### 6.3 吞吐率说明

当前实现属于“单帧迭代处理”结构：

- 一次只处理一个测量向量 `z`。
- `in_ready` 仅在 `ST_IDLE` 时拉高。
- 当前帧完成并进入 `ST_OUTPUT` 后，等待 `out_ready` 接收输出。
- 输出完成后回到 `ST_IDLE`，接收下一帧。

因此本设计不是每拍接收一个新测量的全深流水架构，而是面积更可控、结构更清晰的多周期硬件 IP。该方案适用于传感器采样率不高、需要在 FPGA 中稳定完成滤波计算的应用场景。若后续需要更高吞吐，可进一步将多个矩阵阶段拆成可并行工作的流水级，或复制 MAC 单元提升并行度。

## 7. 异常处理

### 7.1 奇异矩阵检测

当 `S` 矩阵行列式绝对值过小时，认为 `S` 不可逆或数值上接近奇异。模块通过 `DET_EPS` 判断：

```text
abs(det(S)) <= DET_EPS
```

若触发该条件：

- `err_singular` 置位。
- 不继续执行卡尔曼增益计算。
- 进入 `ST_OUTPUT` 输出当前状态。

### 7.2 除法异常

`fixed_div_iter` 提供：

- `div_zero`：分母为 0。
- `overflow`：商超出输出定点范围。

顶层检测到除法异常后同样置位 `err_singular`。

### 7.3 饱和处理

加法、减法、乘法结果写回 32 位定点数据前均经过饱和处理。饱和处理避免了二进制补码溢出后符号反转的问题，提高固定点滤波的可控性。

## 8. 验证说明

### 8.1 已提供 testbench

`tb/tb_kalman_filter_matrix.v` 覆盖以下场景：

1. 复位后使用默认模型输入二维位置测量。
2. 检查正常测量更新时 `err_singular` 不应置位。
3. 检查 `out_ready` 拉低时 `x_out` 保持稳定。
4. 通过配置将 `H` 和 `R` 清零，制造奇异 `S` 场景，检查 `err_singular` 是否置位。
5. 检查输出数量，防止状态机卡死或漏输出。

`tb/tb_kalman_filter_axi.v` 覆盖以下接口场景：

1. AXI-Lite 读取版本寄存器和参数寄存器。
2. AXI-Lite 同拍 AW/W 写入配置寄存器。
3. AXI-Lite 分离 AW/W 写入配置寄存器。
4. AXI-Stream 输入测量数据并等待 AXI-Stream 输出状态向量。
5. 忙状态下写配置空间返回 `SLVERR`。
6. 输出 `TLAST` 和 `TUSER` 信号基本检查。

### 8.2 仿真入口

工程提供 `Makefile`：

```sh
make sim
```

默认调用：

```sh
iverilog -g2001 -Wall -Irtl -o build/tb_kalman_filter_matrix.vvp ...
vvp build/tb_kalman_filter_matrix.vvp
iverilog -g2001 -Wall -Irtl -o build/tb_kalman_filter_axi.vvp ...
vvp build/tb_kalman_filter_axi.vvp
```

### 8.3 当前环境验证状态

当前开发环境已安装 YosysHQ OSS CAD Suite，工具路径为：

```text
~/.local/oss-cad-suite/bin
```

已执行 `make sim`，基础计算核 testbench 和 AXI 封装 testbench 均通过：

```text
PASS: kalman_filter_matrix smoke test completed
PASS: kalman_filter_axi interface smoke test completed
```

已执行 Yosys 前端综合检查，命令流程包括 `read_verilog`、`hierarchy -top kalman_filter_axi`、`proc`、`opt` 和 `stat`。Yosys 能够成功读入 RTL 并完成统计，日志可通过 `make yosys-check` 生成到 `build/yosys-check.log`。Yosys 输出中关于寄存器数组展开的 warning 属于当前小规模矩阵寄存器实现的预期现象，不影响基础综合读入。

针对 Xilinx Artix-7，工程已补充默认 out-of-context timing 约束文件 `constraints/artix7_timing.xdc`，默认器件为 `xc7a35tcsg324-1`，默认时钟为 100 MHz。由于 `kalman_filter_axi` 暴露 AXI-Lite 和 AXI-Stream 宽总线，该模块按 FPGA 内部 IP 处理，不作为直接绑定封装引脚的板级顶层。已执行 `make yosys-artix7`，Yosys `synth_xilinx -family xc7` 能生成 Artix-7 综合网表 `build/kalman_filter_axi_artix7_synth.v`。真正 post-route timing simulation 需要 Vivado 完成布局布线并导出 SDF；工程已提供 `make vivado-artix7` 和 `make vivado-timesim`，但当前机器未安装 Vivado，需在 Vivado 环境中运行。

## 9. 资源与时序特征分析

本设计从结构上避免一次性展开全矩阵运算，资源特征如下：

- 矩阵运算主要通过一条共享 MAC 路径进行多周期复用。
- 中间矩阵使用寄存器数组保存，便于状态机分阶段读写。
- 行列式倒数使用多周期除法器，避免组合除法成为关键路径。
- 默认 `N=4,M=2` 下，矩阵规模固定，控制逻辑简单。

相较于一拍展开式实现，本设计的优势是：

1. DSP 和 LUT 消耗更可控。
2. 关键路径更短。
3. 易于在后续版本中增加流水级寄存器。
4. 可通过复制 MAC 单元在面积和吞吐率之间做工程折中。

需要注意的是，本说明中的资源消耗为结构分析，具体 LUT、FF、DSP、Fmax 数据需在目标 FPGA 器件和综合工具中完成综合后给出。

## 10. 可综合性说明

RTL 使用 Verilog-2001 风格编写，避免使用 SystemVerilog 专有数组端口或高级语法。主要可综合结构包括：

- 同步状态机。
- 寄存器数组。
- 计数器。
- 加法器、减法器、乘法器。
- 移位、比较、饱和逻辑。
- 多周期恢复除法器。

复位和配置阶段存在 `for` 循环，用于初始化寄存器数组；这些循环为静态边界，可由综合工具展开为寄存器复位逻辑。计算阶段没有用多层循环在一拍内展开完整矩阵乘法，而是通过计数器逐拍调度。

## 11. 已知限制与后续改进

当前版本的限制如下：

1. 默认完整支持 `M=2` 的 `2x2` 矩阵求逆；虽然模块保留了 `N/M` 参数，但求逆路径针对 `M=2` 实现。
2. 当前为单帧多周期处理结构，不支持每拍输入一帧测量。
3. AXI-Lite/AXI-Stream 封装为轻量版本，适合 IP 集成原型；若用于复杂 SoC，可继续补充中断、错误计数、软复位和 DMA 描述符支持。

后续可改进方向：

1. 增加 AXI-Lite 软复位、启动模式、中断使能和错误计数寄存器。
2. 增加 AXI-Stream `TKEEP/TID/TDEST` 等可选信号，适配更复杂的数据通路。
3. 复制 2 条或 4 条 MAC 数据通路，提高矩阵运算并行度。
4. 对关键 MAC 路径插入更多寄存器，进一步提升最高工作频率。
5. 增加 Python/Matlab 浮点参考模型，自动对比固定点误差。
6. 在 Vivado、Quartus 或 Yosys 中完成综合报告，补充 LUT、FF、DSP、BRAM 和 Fmax 数据。

## 12. 结论

本项目完成了一套面向 FPGA 的卡尔曼滤波 Verilog IP 初版实现。设计使用 `Q16.16` 固定点数、AXI-Lite 配置接口、AXI-Stream 数据接口和 Joseph 协方差更新形式，默认适配二维位置/速度估计场景。

从硬件实现方式看，当前 RTL 已由早期公式展开式结构调整为计数器调度的多周期 MAC 数据通路，并使用迭代除法器替代组合除法。该结构更符合 FPGA 的资源复用和时序设计习惯，不属于简单复制 CPU/GPU 软件矩阵运算的实现方式。

在结项审核中，可将本工程作为 FPGA 卡尔曼滤波 IP 的可综合 RTL 原型提交。后续若需要进入板级应用或工程部署，建议补充目标器件综合报告、仿真日志、固定点误差对比报告和实际传感器数据测试结果。
