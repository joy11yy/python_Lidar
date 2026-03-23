# 完整示例：先筛选，再按需下载
import requests
from pyGEDI import *
import re

# 1. 登录
session = sessionNASA('2122591542@qq.com', 'Yan13618530568!')

# 2. 设置区域
bbox = [38.2, -123, 37.5, -122.5]

# 3. 获取文件列表
url = f'https://lpdaacsvc.cr.usgs.gov/services/gedifinder?product=GEDI02_A&version=001&bbox={bbox}&output=json'
content = requests.get(url)
files = content.json().get('data')

# 4. 按日期筛选（例如只下载2020年6-8月的）
selected_urls = []
for f in files:
    # 提取日期（URL中通常包含YYYY/MM/DD）
    match = re.search(r'/(\d{4}/\d{2}/\d{2})/', f)
    if match:
        date_str = match.group(1).replace('/', '-')  # 转为 YYYY-MM-DD
        if '2020-06' <= date_str <= '2020-08':  # 只选夏季
            selected_urls.append(f)
            print(f"选择: {date_str} - {f.split('/')[-1]}")

print(f"\n共 {len(selected_urls)} 个文件，预计大小约 {len(selected_urls)*0.5} GB")

# 5. 确认后下载
confirm = input("是否下载？(y/n): ")
if confirm.lower() == 'y':
    for url in selected_urls:
        url_response('data/GEDI02_A.001/', url, session)