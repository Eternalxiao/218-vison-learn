from maix import app

# 设置一些测试配置
app.set_app_config_kv("test_key1", "hello", '0', True)
app.set_app_config_kv("test_key2", 'aaa', '0', True)

# 获取当前应用ID
app_id = app.app_id()
print("Current App ID:", app_id)

# 读取配置文件内容
try:
    with open("/flash/config.json", "r") as f:
        content = f.read()
        print("Config File Content:")
        print(content)
        
        # 检查当前应用的配置是否存在
        if app_id in content:
            print(f"Found configuration for {app_id}")
        else:
            print(f"No configuration found for {app_id}")
except Exception as e:
    print("Error reading config file:", e)

# 直接读取配置值
value = app.get_app_config_kv("test_key1", "default")
print("test_key1 value:", value)