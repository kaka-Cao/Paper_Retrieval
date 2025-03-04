import requests

proxies = {
    "http": "http://127.0.0.1:7890",  # Clash HTTP 代理
    "https": "http://127.0.0.1:7890"
}

try:
    response = requests.get('https://httpbin.org/ip', proxies=proxies, timeout=10)
    print(response.json())
except requests.exceptions.ProxyError as e:
    print(f"Proxy error: {e}")
