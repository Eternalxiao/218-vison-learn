
from maix import touchscreen, camera, display, image, time
from maix.image import Image
import math

class GUI:
    def __init__(self) -> None:
        self.background = None
        self.items      = list()
        self.callbacks  = list()
        self.labels     = list()

        self.touch_x = 0
        self.touch_y = 0

        # 加载并设置中文字体
        image.load_font("sourcehansans", "/maixapp/share/font/SourceHanSansCN-Regular.otf")
        image.set_default_font("sourcehansans")
        #print("fonts:", image.fonts())

        self._ts   = touchscreen.TouchScreen()
        self._disp = display.Display()      # 默认的宽高 -- 552*368
        self._last_pressed = 0  # 记录上次触摸状态（0/1 表示 抬起 / 按下）

    def _is_in_item(self, item_id : int, x : int, y : int) -> bool:
        """
            检查坐标(x,y)是否在指定界面元素( item_id )区域内
        """
        if item_id >= len(self.items) or self.background == None:
            return False
        
        # 获取元素在显示设备上的实际坐标（考虑不使用默认分辨率的适配）
        item_pos = self.items[item_id]
        item_disp_pos = image.resize_map_pos(
            self.background.width(), self.background.height(),  # w_in, h_in -- 图像的宽高
            self._disp.width(), self._disp.height(),    # w_out, h_out -- 屏幕的大小即552*224
            image.Fit.FIT_CONTAIN,  # resize method -- 保持长宽比，用黑色填充空白区域
            item_pos[0], item_pos[1], item_pos[2], item_pos[3]  # 原点
        )
        
        # 进行边界坐标检查
        if x > item_disp_pos[0] and x < (item_disp_pos[0]+item_disp_pos[2]) and y > item_disp_pos[1] and y < (item_disp_pos[1]+item_disp_pos[3]):
            return True
        else:
            return False

    def createButton(self, x : int, y:int, width : int, height : int) -> int:
        """
            创建按钮组件
            参数: x,y - 左上角坐标, width/height - 宽高
            返回: 新元素的ID
        """
        item_id = len(self.items)   # 新元素ID为当前元素数量
        self.items.append([x, y, width, height])    # 存储位置信息
        self.callbacks.append(None) # 初始化回调的函数
        self.labels.append(None)    # 初始化标签值
        return item_id

    def setItemCallback(self, item_id: int, cbf ) -> None:
        """
            设置界面组件的回调函数
            参数: 
            item_id - 目标元素ID 
            cbf - 回调函数，格式: cbf(item_id:int, state:int)
        """
        if item_id >= len(self.items):
            return
        self.callbacks[item_id] = cbf   # 绑定回调函数

    def setItemLabel(self, item_id:int, label:str) -> None:
        '''
        设置界面组件中所显示信息
        '''
        if item_id >= len(self.items):
            return
        self.labels[item_id] = label

    def get_touch(self) -> tuple:
        '''
            返回最近时间触摸动作的位置（反向映射）
        '''
        if self.background == None:
            return (0,0)
        
        # 将屏幕坐标(520*324)转换为屏幕显示图像的坐标 -- 考虑不使用默认分辨率的适配
        x, y = image.resize_map_pos_reverse(
            self.background.width(), self.background.height(),  # w_in, h_in -- 屏幕显示图像的
            self._disp.width(), self._disp.height(),    # w_out, h_out -- 屏幕初始化时的
            image.Fit.FIT_CONTAIN, 
            self.touch_x, self.touch_y
        )
        return (max(x, 0), max(y, 0))  # 确保非负坐标

    def run(self, background:Image) -> None:
        """
            检查图像的点击输入，并对传入的拍摄图像进行按键 ui 的添加
        """
        self.background = background
        
        # 获得默认屏幕的点击坐标（520*324）
        self.touch_x, self.touch_y, pressed = self._ts.read()   # 得到按下的x y 按下的状态(长按和点击)
        
        # 检测输入的点击
        if self._last_pressed != pressed:
            self._last_pressed = pressed

            # 如果和上次触摸的位置不一样 -- 逐按钮判断点击的是否为规定区域
            for id in range(len(self.items)):
                if self._is_in_item(id, self.touch_x, self.touch_y):    # 判断点击的是否为规定区域
                    if self.callbacks[id] != None:
                        self.callbacks[id](id, pressed)     # 根据对应的回调函数进行回调
                    break

        # 更新界面元素
        for id in range(len(self.items)):
            label_size = image.string_size(self.labels[id]) # 根据id获取渲染文本的宽度和高度

            # 确定标识内容的（x, y）位置 -- 如果按钮的大小够就 居中 否则 左起
            # self.item[[x, y, w, h], ....]                                             # 如果 按钮的宽大于文本的宽(高) 否则使用按钮的x(y)  -- 居中/左起
            label_x = (self.items[id][0] + (self.items[id][2] - label_size.width())//2) if self.items[id][2] > label_size.width() else self.items[id][0]
            label_y = (self.items[id][1] + (self.items[id][3] - label_size.height())//2) if self.items[id][3] > label_size.height() else self.items[id][1]

            # rect 按钮框
            self.background.draw_rect(self.items[id][0], self.items[id][1], self.items[id][2], self.items[id][3], image.COLOR_WHITE, 2)
            
            # 标识文本
            if self.labels[id] != None:
                self.background.draw_string(label_x, label_y, self.labels[id], image.COLOR_BLACK)

        self._disp.show(self.background)
        fps = time.fps()
        print(f"time: {1000/fps:.02f}ms, fps: {fps:.02f}\n")

    def show(self, image):
        self._disp.show(image)
        
if __name__ == '__main__':
    
    def btn_pressed(btn_id, state): # 回调函数
        print('item {} state: {}'.format(btn_id, state))

    disp_width  = 320
    disp_height = 224
    cam = camera.Camera(disp_width, disp_height)   # Manually set resolution, default is too large
    
    gui = GUI()
    btn_id1= gui.createButton(0,0,60,40)
    gui.setItemLabel(btn_id1, '< exit')
    gui.setItemCallback(btn_id1, btn_pressed)

    btn_id2= gui.createButton(0,disp_height-40,60,40)
    gui.setItemLabel(btn_id2, '试试')
    gui.setItemCallback(btn_id2, btn_pressed)

    btn_id3= gui.createButton(disp_width-60,disp_height-40,60,40)
    gui.setItemLabel(btn_id3, 'GameOver')
    gui.setItemCallback(btn_id3, btn_pressed)

    btn_id4= gui.createButton(disp_width-60,0,60,40)
    gui.setItemLabel(btn_id4, 'sad')
    gui.setItemCallback(btn_id4, btn_pressed)

    while True:
        img = cam.read()
        gui.run(img)
        time.sleep(0.1)
    
