"""
声光提示 IO - MaixCAM-Pro 板载照明LED(B3脚) + 喇叭(alert.wav)
- LED: 支持连续闪烁(GPIO toggle), 高电平点亮, 参考官方wiki示例
- 喇叭: MaixCAM-Pro 内置 PA+1W喇叭, maix.audio.Player 播放WAV
- 全部 try/except: 任何硬件初始化失败都静默降级, 绝不影响主流程
"""


class AlertIO:

    def __init__(self, cfg):
        a = cfg.get("alert", {})
        self.led_enabled = a.get("led_enabled", True)
        self.wav_path = a.get("wav", "alert.wav")
        self.volume = a.get("volume", 95)
        self.led = None
        self.player = None
        self._init_led()
        self._init_audio()
        print("[alert] led=%s audio=%s" % (
            "ok" if self.led else "-", "ok" if self.player else "-"))

    def _init_led(self):
        """照明LED: MaixCAM-Pro=B3, MaixCAM2=B25 (官方wiki示例引脚)"""
        if not self.led_enabled:
            return
        try:
            from maix import gpio, pinmap, sys
            if sys.device_id() == "maixcam2":
                pin_name, gpio_name = "B25", "GPIOB25"
            else:
                pin_name, gpio_name = "B3", "GPIOB3"
            pinmap.set_pin_function(pin_name, gpio_name)
            self.led = gpio.GPIO(gpio_name, gpio.Mode.OUT)
            self.led.value(0)
        except Exception as e:
            print("[alert] led init fail:", e)
            self.led = None

    def _init_audio(self):
        """喇叭: WAV文件自动识别采样率/格式, 官方推荐48kHz/单声道/S16_LE"""
        try:
            import os
            if not os.path.exists(self.wav_path):
                print("[alert] wav missing:", self.wav_path)
                return
            from maix import audio
            self.player = audio.Player(self.wav_path)
            self.player.volume(self.volume)
        except Exception as e:
            print("[alert] audio init fail:", e)
            self.player = None

    def led_toggle(self):
        if self.led is not None:
            try:
                self.led.toggle()
            except Exception:
                pass

    def led_off(self):
        if self.led is not None:
            try:
                self.led.value(0)
            except Exception:
                pass

    def play_once(self):
        """播放一次提示音 (play只阻塞到写完缓冲, 实际出声异步)"""
        if self.player is not None:
            try:
                self.player.play()
            except Exception as e:
                print("[alert] play fail:", e)