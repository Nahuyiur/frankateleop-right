# GELLO 双臂 delta EEF（.98 左臂，.86 右臂）

入口：`bash /home/pnp/frankateleop/A_run_dual_gello_eef.sh`。
无参数只显示帮助，不启动硬件。

## 侧装坐标

先用每个 GELLO 串口在 `PORT_CONFIG_MAP` 内的 joint offsets/signs 得到主臂关节角，
再用 Franka 模型 FK 得到 **主臂基座系** 的末端位姿。侧装不改变这一步的 FK。
必须确认同构结构/零点/符号和 FK 的末端定义对应；锚点不能修复错误的 FK 或工具定义。

设 `C = R_followerBase_leaderBase = R_world_followerBase^-1 * R_world_leaderBase`。
每侧参数 `leader_to_follower_base_rpy_deg` 指定 C：SciPy `xyz` 外禀旋转、单位度。
只有同一对主从的基座轴向确实一致时 C=I，即 `[0,0,0]`。
两者都侧装不充分；也不能给两侧一律加 90 度或镜像关节符号。

映射（每次启动独立锁定主从锚点）：

```
p_target = p_F0 + translation_scale * C * (p_L - p_L0)
delta_R_F = C * R_L * inverse(R_L0) * inverse(C)
R_target = Exp(rotation_scale * Log(delta_R_F)) * R_F0
```

delta 是相对锚点的累计变化，不是逐帧相加。没有基座平移项，因为常量平移在位置差中抵消。
旋转在 SO(3) 上处理，不直接减 Euler 角。旧实现用末端锚点旋转代替 C，主从末端姿态
不同会使平移方向随锚点改变；CLI 现在显式使用物理基座映射。
左、右同向物理运动都映射到各自 follower base，不要求左右数值 delta 完全相同。
本实现不是共用世界坐标的双臂碰撞规划器，也没有双臂碰撞检测。

## 配置和启动

配置：`/home/pnp/frankateleop/config/gello_eef_dual.json`。
出厂模板中 `calibration_verified=false`，`workspace_base_m=null`，仅允许预览。
不得为了启动而盲目改成 true：请先确认每侧基座旋转、主臂 FK/末端定义、运动方向，
并填写每侧 **Franka 基座系** 的绝对工作空间（不是世界系/不是 GELLO 基座系）：

```
"workspace_base_m": {"x": [xmin, xmax], "y": [ymin, ymax], "z": [zmin, zmax]}
```

侧装下 z 可以为负；不要照搬桌面安装的正 z 区间。每侧还受 0.12 m / 0.6 rad
锚点包络限制，默认平移/旋转比例均为 0.5。

```bash
cd /home/pnp/frankateleop
bash A_run_dual_gello_eef.sh --check    # 本机无硬件测试，无 SSH
bash A_run_dual_gello_eef.sh --plan     # 只打印左右命令
bash A_run_dual_gello_eef.sh --preview  # 会启动硬件服务/初始化主臂，但不发送 EEF/夹爪目标
# 工作区安全、FCI 就绪，确认并填写上述配置后：
bash A_run_dual_gello_eef.sh --execute
```

可用 `--config /绝对路径/配置.json` 指定配置。先 Ctrl+C 完全退出预览，再运行 execute。
右臂使用现有 `.98 -> .86` SSH key，不保存密码。启动前比较两端适配器文件摘要；
不同则拒绝运行，不自动覆盖代码。不存在自动杀端口/替换旧控制栈的行为。
右侧入口 `B_run_right_gello_eef.sh` 是 **在 .86 本地运行** 的单臂脚本。
旧左侧入口仍在，但 live 需显式 `--base-alignment-rpy-deg R P Y --calibration-verified`。

## 双臂运行边界

每台机器复用自己的 robot/gripper 1–2 脚本、无回家/无自动开夹爪的 EEF 节点、
同一份 GELLO adapter。左右 serial/offset/sign 配置不互换。
右阻尼目标沿用当前官方右 4_run_env.sh 实际使用的通用目标，不复制左臂目标或未使用的
fr3_right 目标；此改动不重标定主臂参数。

两个 adapter 均就绪后，协调进程才发送 GO；各自倒计时结束后重新记录主从锚点。
两臂不是硬实时同步采样。任一侧出错/HOLD/退出，协调器结束两侧；SSH 中断、协调器退出、
STOP、stdin EOF 或 1 秒 lease 超时会使 adapter 退出并停止继续更新目标。
这不是 1 秒硬实时急停保证：设备/RPC 调用可能阻塞，Cartesian 控制仍可能向最后目标收敛。
Ctrl+C 清理本次启动的服务；硬件急停、低层关节限位/碰撞保护和实际重力安装配置仍必需。
脚本不修改 Desk/低层重力参数。低层安装/重力必须已经与侧装实物匹配。

## 验证范围

单元测试覆盖同向侧装无需额外旋转、不同安装旋转、非同姿态锚点、位置/旋转物理方向、
重复累计 delta 不二次积分、左右配置分离、未标定禁止执行、HOLD 传播、双 ready gate、
单侧退出联停、就绪超时和 lease 失效。测试使用数学/假进程，不控制真实硬件。
参数标定与低速真机验收仍由现场完成。
