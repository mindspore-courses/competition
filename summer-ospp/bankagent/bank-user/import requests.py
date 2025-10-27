import requests
from requests.exceptions import RequestException

# 测试 URL
TEST_URL = "https://sis-ext.cn-north-4.myhuaweicloud.com/v1"  # 华为云 North4 API 根地址

# 代理列表示例
# 格式: {"type": "http" or "socks5", "address": "127.0.0.1:端口"}
proxies_to_test = [
    {"type": "http", "address": "8.156.67.245:3128"},
    {"type": "socks5", "address": "127.0.0.1:1080"},
]

def test_proxy(proxy):
    proxy_type = proxy["type"]
    proxy_address = proxy["address"]

    if proxy_type == "http":
        proxies = {
            "http": f"http://{proxy_address}",
            "https": f"http://{proxy_address}",
        }
    elif proxy_type == "socks5":
        proxies = {
            "http": f"socks5h://{proxy_address}",
            "https": f"socks5h://{proxy_address}",
        }
    else:
        print(f"[!] 未知代理类型: {proxy_type}")
        return

    try:
        resp = requests.get(TEST_URL, proxies=proxies, timeout=10)
        if resp.status_code == 401 or resp.status_code == 403:
            # 未认证，但能到达服务器
            print(f"[✅] {proxy_type.upper()} {proxy_address} 可以访问华为云 North4（未认证）")
        else:
            print(f"[✅] {proxy_type.upper()} {proxy_address} 可以访问，HTTP状态码: {resp.status_code}")
    except RequestException as e:
        print(f"[❌] {proxy_type.upper()} {proxy_address} 无法访问: {e}")

if __name__ == "__main__":
    for proxy in proxies_to_test:
        test_proxy(proxy)
