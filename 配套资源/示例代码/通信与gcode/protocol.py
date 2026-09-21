"""文本、JSON 与带帧头的二进制视觉数据教学示例。"""

import json
import struct


HEADER = b"\xAA\x55"          # 帧头
TAIL = b"\r\n"                # 帧尾
MAX_BODY_LENGTH = 4096        # 体最大长度


def encode_text(class_id, u, v, angle):
    """编码为文本(CSV)格式。"""
    return f"{int(class_id)},{int(u)},{int(v)},{float(angle):.2f}\r\n".encode("ascii")


def decode_text(data):
    """解码文本(CSV)格式。"""
    class_id, u, v, angle = data.decode("ascii").strip().split(",")
    return {"class_id": int(class_id), "u": int(u), "v": int(v), "angle": float(angle)}


def encode_json(class_id, u, v, angle):
    """编码为 JSON 格式。"""
    payload = {"class_id": int(class_id), "u": int(u), "v": int(v), "angle": float(angle)}
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def decode_json(data):
    """解码 JSON 格式。"""
    return json.loads(data.decode("utf-8"))


def encode_binary(command, payload, endian="big"):
    """编码为带帧头的二进制数据。

    帧结构: 帧头(2) + 体长度(2) + 体(命令1 + 载荷N) + 校验和(1) + 帧尾(2)
    """
    if not 0 <= command <= 255:
        raise ValueError("命令必须是单字节(0~255)")
    if not isinstance(payload, bytes):
        raise TypeError("载荷必须是 bytes 类型")
    body = bytes([command]) + payload
    if len(body) > MAX_BODY_LENGTH:
        raise ValueError("载荷过大")
    order = ">" if endian == "big" else "<" if endian == "little" else None
    if order is None:
        raise ValueError("字节序必须是 big 或 little")
    checksum = sum(body) & 0xFF
    return HEADER + struct.pack(order + "H", len(body)) + body + bytes([checksum]) + TAIL


class BinaryFrameParser:
    """增量解析带帧头的数据，可处理分段接收和连续到达的帧。"""

    def __init__(self, endian="big"):
        if endian not in {"big", "little"}:
            raise ValueError("字节序必须是 big 或 little")
        self.order = ">" if endian == "big" else "<"
        self.buffer = bytearray()

    def feed(self, chunk):
        """喂入新收到的字节块，返回本次解析出的所有帧列表。"""
        self.buffer.extend(chunk)
        frames = []
        while True:
            # 1. 找帧头
            start = self.buffer.find(HEADER)
            if start < 0:
                # 未找到帧头：若缓冲区末尾可能是半个帧头则保留最后一字节
                self.buffer[:] = self.buffer[-1:] if self.buffer.endswith(HEADER[:1]) else b""
                break
            if start:
                # 丢弃帧头之前的垃圾数据
                del self.buffer[:start]
            if len(self.buffer) < 4:
                break  # 长度字段还没收全，等下次

            # 2. 读取体长度
            body_length = struct.unpack(self.order + "H", self.buffer[2:4])[0]
            if body_length < 1 or body_length > MAX_BODY_LENGTH:
                # 长度非法：跳过当前帧头首字节，重新寻找
                del self.buffer[0]
                continue

            total_length = 2 + 2 + body_length + 1 + len(TAIL)
            if len(self.buffer) < total_length:
                break  # 帧还没收完，等下次

            candidate = bytes(self.buffer[:total_length])

            # 3. 校验帧尾
            if candidate[-len(TAIL):] != TAIL:
                del self.buffer[0]
                continue

            body = candidate[4 : 4 + body_length]
            expected = candidate[4 + body_length]

            # 4. 校验和
            if (sum(body) & 0xFF) != expected:
                del self.buffer[0]
                continue

            # 5. 解析成功
            frames.append((body[0], body[1:]))
            del self.buffer[:total_length]
        return frames


def demo():
    """演示三种编码方式，并验证解析器能处理分段数据。"""
    values = (1, 320, 224, 15.5)

    # 文本格式
    print(encode_text(*values).decode("ascii"), end="")

    # JSON 格式
    print(encode_json(*values).decode("utf-8"), end="")

    # 二进制格式: 载荷用 >Bhhf 打包(1字节命令 + 2字节u + 2字节v + 4字节角度)
    payload = struct.pack(">Bhhf", *values)
    frame = encode_binary(0x01, payload)

    # 故意拆成两段喂给解析器，验证增量解析能力
    parser = BinaryFrameParser()
    decoded = parser.feed(frame[:4]) + parser.feed(frame[4:])
    print(decoded)


if __name__ == "__main__":
    demo()