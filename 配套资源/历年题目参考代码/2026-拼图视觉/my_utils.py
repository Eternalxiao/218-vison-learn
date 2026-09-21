"""
串口管理 - 调用 gcode_generator 生成G代码, 通过UART发送
流控v2 (时间滴灌 drip-feed, 最大容错):
- 每个块(G1+M400 / M280+G4 ...)先估算执行时间:
    G1 按 距离/速度, G4 按暂停时长, M280 按舵机角度差
- 发一块 -> 等待 "估算时间 x time_margin + min_block_ms" -> 再发下一块
- 效果: 下位机缓冲区任何时刻 <= 1 块, 再小的缓冲也不会爆
- tick_cb回调: 发送期间主程序可刷新屏幕/查deadline, 返回False中止发送
- pace_by_time=false 时退回旧模式(仅block_delay_ms块间延时)
"""
from maix import uart, time
import gcode_generator


class MyUtils:
    def __init__(self, cfg):
        self.cfg = cfg
        self.port = None
        self.gcode = gcode_generator.GCodeGenerator(cfg)
        self.sync_cmds = ("M400", "G4")

        self._load_serial_cfg(cfg)

        # 运动状态跟踪 (估算执行时间用)
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_z = 0.0
        self.last_f = cfg["mechanical"].get("feed_rate", 3000)
        self.last_servo = cfg["mechanical"].get("e_home", 90.0)

        s = cfg["serial"]
        if s["enabled"]:
            print("\n[serial] enabled")
            try:
                self.port = uart.UART(s["device"], s["baudrate"])
                print("[serial] %s @ %d" % (s["device"], s["baudrate"]))
            except Exception as e:
                print("[serial] fail:", e)
                self.port = None
        else:
            print("\n[serial] disabled (sim mode)")

    def _load_serial_cfg(self, cfg):
        s = cfg["serial"]
        self.block_delay_ms = s.get("block_delay_ms", 100)
        # ===== 时间滴灌参数 =====
        self.pace_by_time = s.get("pace_by_time", True)
        self.time_margin = s.get("time_margin", 1.5)        # 估算时间安全系数
        self.min_block_ms = s.get("min_block_ms", 400)      # 每块最少等待(ms)
        self.servo_ms_per_deg = s.get("servo_ms_per_deg", 5)  # 舵机每度毫秒数(估)

    # ==================== 对外接口 ====================

    def send_init(self, tick_cb=None):
        """上电初始化"""
        self.gcode.clear()
        self.gcode.start_gcode()
        return self._send(tick_cb)

    def send_piece(self, result, tick_cb=None):
        """发送一个碎片的完整动作 (滴灌节奏, 全程约15-20秒)"""
        self.gcode.clear()
        self.gcode.pick_place_rotate(
            result["from_px"],
            result["target_px"],
            result["theta"]
        )
        return self._send(tick_cb)

    def finish(self, tick_cb=None):
        """所有碎片完成: 归位"""
        self.gcode.clear()
        self.gcode.end_gcode()
        return self._send(tick_cb)

    def send_calib_move(self, tick_cb=None):
        """Step3标定: 移动到offset位置"""
        self.gcode.clear()
        self.gcode.calib_move()
        return self._send(tick_cb)

    def send_calib_point(self, u, v, tick_cb=None):
        """PNT多点校验: 移动到像素(u,v)对应位置"""
        self.gcode.clear()
        self.gcode.calib_point(u, v)
        return self._send(tick_cb)

    def reload_cfg(self, cfg):
        """现场标定后重新加载配置"""
        self.cfg = cfg
        self.gcode = gcode_generator.GCodeGenerator(cfg)
        self._load_serial_cfg(cfg)
        self.last_f = cfg["mechanical"].get("feed_rate", 3000)

    # ==================== 块切分 ====================

    def _split_safe_blocks(self, lines):
        """
        把G代码按同步指令(M400/G4)切块。
        保证: 绝不把移动指令和它后面的M400拆开, 每块都是完整语义单元。
        """
        blocks = []
        cur = []
        for line in lines:
            cur.append(line)
            tokens = line.split()
            cmd = tokens[0].upper() if tokens else ""
            if cmd in self.sync_cmds:
                blocks.append(cur)
                cur = []
        if cur:
            blocks.append(cur)
        return blocks

    # ==================== 执行时间估算 ====================

    def _estimate_block_s(self, block):
        """估算一个块的物理执行时间(秒), 同时更新位置/舵机状态"""
        t = 0.0
        for line in block:
            tokens = line.split()
            if not tokens:
                continue
            cmd = tokens[0].upper()
            if cmd == "G1":
                x, y, z, f = None, None, None, None
                for tok in tokens[1:]:
                    try:
                        v = float(tok[1:])
                    except Exception:
                        continue
                    c = tok[0].upper()
                    if c == "X":
                        x = v
                    elif c == "Y":
                        y = v
                    elif c == "Z":
                        z = v
                    elif c == "F":
                        f = v
                if f is not None:
                    self.last_f = f
                feed = max(1.0, self.last_f / 60.0)  # mm/s
                nx = self.last_x if x is None else x
                ny = self.last_y if y is None else y
                nz = self.last_z if z is None else z
                dist = ((nx - self.last_x) ** 2 + (ny - self.last_y) ** 2 +
                        (nz - self.last_z) ** 2) ** 0.5
                t += dist / feed
                self.last_x, self.last_y, self.last_z = nx, ny, nz
            elif cmd == "G92":
                # 原点设定: 只更新位置状态, 不计运动时间
                for tok in tokens[1:]:
                    try:
                        v = float(tok[1:])
                    except Exception:
                        continue
                    c = tok[0].upper()
                    if c == "X":
                        self.last_x = v
                    elif c == "Y":
                        self.last_y = v
                    elif c == "Z":
                        self.last_z = v
                t += 0.05
            elif cmd == "G4":
                for tok in tokens[1:]:
                    if tok[0].upper() == "P":
                        try:
                            t += float(tok[1:]) / 1000.0
                        except Exception:
                            pass
            elif cmd == "M280":
                for tok in tokens[1:]:
                    if tok[0].upper() == "S":
                        try:
                            ang = float(tok[1:])
                        except Exception:
                            continue
                        t += abs(ang - self.last_servo) * self.servo_ms_per_deg / 1000.0
                        self.last_servo = ang
            elif cmd in ("M106", "M107"):
                t += 0.05
            else:
                t += 0.02
        return t

    # ==================== 滴灌等待 ====================

    def _paced_sleep(self, seconds, tick_cb, bi, nb):
        """分段sleep(200ms), 每段调tick_cb刷新屏幕; tick_cb返回False则中止"""
        t_end = time.ticks_ms() + int(seconds * 1000)
        while time.ticks_ms() < t_end:
            if tick_cb is not None:
                if not tick_cb(bi, nb):
                    return False
            time.sleep_ms(200)
        return True

    # ==================== 串口发送 ====================

    def _send(self, tick_cb=None):
        """滴灌发送: 每块之间按其估算执行时间等待, 下位机缓冲始终<=1块"""
        lines = self.gcode.gcode_lines
        if not lines:
            print("[gcode] empty")
            return False

        if self.port is None:
            for line in lines:
                print("  [sim]", line)
            self.gcode.clear()
            return True

        try:
            blocks = self._split_safe_blocks(lines)
            nb = len(blocks)
            for bi, blk in enumerate(blocks):
                est = self._estimate_block_s(blk)
                self.port.write_str("EEEF\r\n")
                for line in blk:
                    self.port.write_str(line + "\r\n")
                self.port.write_str("FFFE\r\n")
                # 最后一块不等(settle阶段会等), 其余块按估算时间滴灌
                if bi < nb - 1:
                    if self.pace_by_time:
                        wait_s = est * self.time_margin + self.min_block_ms / 1000.0
                    else:
                        wait_s = self.block_delay_ms / 1000.0
                    if not self._paced_sleep(wait_s, tick_cb, bi + 1, nb):
                        print("[gcode] ABORT at block %d/%d" % (bi + 1, nb))
                        self.gcode.clear()
                        return False
            print("[gcode] sent %d lines / %d blocks (paced=%s)" % (
                len(lines), nb, self.pace_by_time))
            self.gcode.clear()
            return True
        except Exception as e:
            print("[gcode] send fail:", e)
            return False