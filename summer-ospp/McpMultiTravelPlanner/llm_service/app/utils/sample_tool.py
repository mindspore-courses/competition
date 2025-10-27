
import random
import string


def generate_unique_str(length: int) -> str:
    """
    生成单个不重复字符的随机字符串（数字+大小写字母）
    :param length: 字符串长度（最大支持 62，因数字+大小写字母共 10+26+26=62 个不重复字符）
    :return: 不重复的随机字符串
    """
    # 1. 定义字符池：数字（0-9） + 大写字母（A-Z） + 小写字母（a-z）
    char_pool = string.digits + string.ascii_uppercase + string.ascii_lowercase
    
    # 2. 校验长度：避免超过字符池总数导致无法生成
    if length > len(char_pool):
        raise ValueError(f"长度不能超过 {len(char_pool)}（字符池共 62 个不重复字符）")
    
    # 3. 随机选择指定数量的不重复字符（sample 方法确保无重复）
    unique_chars = random.sample(char_pool, k=length)
    
    # 4. 将字符列表拼接为字符串
    return ''.join(unique_chars)