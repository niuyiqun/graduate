# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：down.py
@Author  ：niu
@Date    ：2025/12/5 16:50 
@Desc    ：
"""

import os
from huggingface_hub import snapshot_download

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

repo_id = "sentence-transformers/all-MiniLM-L6-v2"

local_dir = "./all-MiniLM-L6-v2"   #
os.makedirs(local_dir, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,          # 断点续传
)
print("下载完成，保存在：", local_dir)

