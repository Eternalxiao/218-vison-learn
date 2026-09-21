
from maix import app, uart, pinmap, time
import struct

# ports = uart.list_devices()

# pinmap.set_pin_function("A16", "UART0_TX")
# pinmap.set_pin_function("A17", "UART0_RX")
device = "/dev/ttyS0"

serial0 = uart.UART(device, 115200)


data = "G1 X-2.588 Y68.729 Z68.000 "
serial0.write_str(data)
print("sent:", data)

print("now wait receive data:")
while not app.need_exit():
    data = serial0.read()
    if data:
        print("Received, type: {}, len: {}, data: {}".format(type(data), len(data), data))
        serial0.write(data)

    time.sleep_ms(1) # sleep 1ms to make CPU free





# G1 X7.593 Y80.505
# M400
# G1 Z70.000
# M400
# M150 U255
# G4 P500
# G1 Z68.000
# G1 X41.418 Y84.347
# M400
# M150 U0
# G4 P500
# G1 Z60.0
# M400
# G28 XY

# on_received start
# 开始解析
# 收到信息all_str_data=4，开始字节提取解析
# 得到指令：order={'four': [1]}
# 执行four, start
# 开始映射白色棋子...
# 开始映射黑色棋子...
# this
#  | | 
# -----
#  |O| 
# -----
#  | |X
# -----
# AI正在思考...
# ai下棋到：move=(2, 0)
# 执行four, end
# 批注from, to：from: [33.0, 15.0]
# 批注from, to：to: (130, 151)
# 启动进行串口发送
# 对像素数数据new_data={'from': (33.0, 15.0), 'to': (130, 151)}进行g代码生成并发送
# 将像素坐标 (u, v) 转换为实际坐标 (x, y, z): u=33.0,v=15.0,x=27.84375,y=12.388392857142858,z=0.0
# 准备逆向解析世界坐标x=27.84375,y=12.388392857142858
# 添加G代码：G1 X-2.588 Y68.729 Z68.000
# 将像素坐标 (u, v) 转换为实际坐标 (x, y, z): u=130,v=151,x=109.6875,y=124.70982142857143,z=0.0
# 准备逆向解析世界坐标x=109.6875,y=124.70982142857143
# 添加G代码：G1 X42.118 Y100.933 Z70.000
# 发送g代码

# G1 X-2.588 Y68.729
# M400
# G1 Z70.000
# M400
# M150 U255
# G4 P500
# G1 Z68.000
# G1 X42.118 Y100.933
# M400
# M150 U0
# G4 P500
# G1 Z60.0
# M400
# G28 XY

