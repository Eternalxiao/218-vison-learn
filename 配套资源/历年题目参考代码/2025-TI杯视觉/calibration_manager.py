import re
import os

class CalibrationManager:
    def __init__(self, constants_file_path="constants.py"):
        self.constants_file_path = constants_file_path
        self.default_distance = 1000.0
    
    def update_constants_file(self, updates):
        """
        更新 constants.py 文件中的常量
        
        Args:
            updates (dict): 需要更新的常量字典 {常量名: 新值}
        
        Returns:
            bool: 更新是否成功
        """
        if not os.path.exists(self.constants_file_path):
            print(f"常量文件不存在: {self.constants_file_path}")
            return False
        
        try:
            # 读取原文件内容
            with open(self.constants_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # 更新常量值
            for constant_name, new_value in updates.items():
                pattern = rf'^({constant_name}\s*=\s*)[^\s#]*(.*)$'
                replacement = rf'\g<1>{new_value}\g<2>'
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            
            # 写回文件
            with open(self.constants_file_path, 'w', encoding='utf-8') as file:
                file.write(content)
            
            print(f"标定完成，已更新常量: {list(updates.keys())}")
            return True
            
        except Exception as e:
            print(f"更新常量文件失败: {e}")
            return False

