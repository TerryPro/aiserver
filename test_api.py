import requests
import json
import re

# 使用本地运行的Jupyter服务器地址
# 根据jupyter_config.py，token设置为空字符串
base_url = "http://localhost:8888"
token = ""  # 从jupyter_config.py中看到token设置为空字符串
# 在URL中添加token参数
base_url_with_token = f"{base_url}/?token={token}" if token else base_url
generate_url = f"{base_url}/aiserver/generate"

# 准备请求数据
data = {
    "intent": "创建一个打印Hello World的函数",
    "language": "python"
}

try:
    # 创建一个会话以保持cookie
    session = requests.Session()
    
    # 首先发送GET请求到首页以获取XSRF令牌和cookie
    response = session.get(f"{base_url_with_token}/lab")
    
    # 从响应中提取XSRF令牌
    xsrf_token = None
    if response.status_code == 200:
        # 尝试从cookie或HTML中获取XSRF令牌
        # 从cookie中提取
        if '_xsrf' in session.cookies:
            xsrf_token = session.cookies['_xsrf']
        # 尝试从HTML中提取
        else:
            xsrf_match = re.search(r'_xsrf.*?value=["\'](.*?)["\']', response.text)
            if xsrf_match:
                xsrf_token = xsrf_match.group(1)
    
    print(f"获取到的XSRF令牌: {xsrf_token}")
    
    # 设置请求头
    headers = {
        "Content-Type": "application/json"
    }
    
    # 添加XSRF令牌到请求头或数据
    if xsrf_token:
        # 方法1: 添加到请求头
        headers["X-XSRFToken"] = xsrf_token
        # 方法2: 添加到请求数据（对于某些框架，这是必需的）
        data["_xsrf"] = xsrf_token
    
    # 发送POST请求，确保URL包含token
    generate_url_with_token = f"{generate_url}?token={token}" if token else generate_url
    response = session.post(generate_url_with_token, headers=headers, json=data)
    
    # 输出响应
    print(f"状态码: {response.status_code}")
    print(f"响应内容: {response.text}")
    
    # 如果请求失败，尝试使用form数据格式（某些服务器可能需要）
    if response.status_code != 200:
        print("尝试使用form数据格式...")
        # 重置数据，确保包含XSRF令牌
        form_data = data.copy()
        if xsrf_token and "_xsrf" not in form_data:
            form_data["_xsrf"] = xsrf_token
        
        # 发送表单数据格式的请求，确保URL包含token
        response = session.post(generate_url_with_token, headers={}, data=form_data)
        print(f"表单数据请求状态码: {response.status_code}")
        print(f"表单数据响应内容: {response.text}")
        
except Exception as e:
    print(f"错误: {e}")