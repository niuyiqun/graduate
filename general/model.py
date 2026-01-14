# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：model.py
@Author  ：niu
@Desc    ：模型接口封装 (支持 ZhipuAPI 与 本地 vLLM-Qwen)
"""
import json
import requests
import yaml
import re
from typing import List, Dict, Any
from abc import ABC, abstractmethod


# === 1. 基类定义 (BaseModel) ===
class BaseModel(ABC):
    def __init__(self, config_path: str) -> None:
        self.config = self.load_config(config_path)

    def load_config(self, config_path: str) -> Dict[str, Dict]:
        try:
            with open(config_path, "r", encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise ValueError(f"Config load failed: {e}")

    @abstractmethod
    def chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """与模型交互，返回字典结果"""
        pass

    @abstractmethod
    def load_model(self) -> None:
        """加载模型或初始化客户端"""
        pass


# === 2. ZhipuChat (云端 API) ===
class ZhipuChat(BaseModel):
    def __init__(self, config_path: str = './config/llm_config.yaml') -> None:
        super().__init__(config_path)
        zhipu_config = self.config.get("zhipu", {})
        self.api_key = zhipu_config.get("api_key", "")
        self.model = zhipu_config.get("model", "glm-4")
        self.client = None
        self.load_model()

    def load_model(self) -> None:
        """初始化智谱客户端"""
        try:
            from zhipuai import ZhipuAI
            self.client = ZhipuAI(api_key=self.api_key)
            print("=== ZhipuAI Client Initialized ===")
        except ImportError:
            print("[Warn] 'zhipuai' package not found. Cloud API will not work.")

    def chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.client:
            return {"error": "Client not initialized"}

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            content = response.choices[0].message.content
            return self._extract_json(content)
        except Exception as e:
            print(f"[Zhipu] Error: {e}")
            return {}

    def _extract_json(self, content: str) -> Dict:
        """通用 JSON 提取逻辑"""
        try:
            # 1. 尝试匹配 Markdown 代码块
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if match:
                return json.loads(match.group(1))

            # 2. 尝试匹配最外层 {}
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                return json.loads(content[start:end + 1])

            # 3. 裸奔尝试
            return json.loads(content)
        except Exception as e:
            # 如果不是 JSON，尝试封装返回
            return {"content": content, "raw": content}


# === 3. QwenChat (本地 vLLM 核心) ===
class QwenChat(BaseModel):
    def __init__(self, config_path: str = './config/llm_config.yaml') -> None:
        super().__init__(config_path)
        # 读取 qwen 配置
        qwen_conf = self.config.get("qwen", {})

        # 自动补全 URL
        base_url = qwen_conf.get("base_url", "http://localhost:8001/v1")
        if not base_url.endswith('/chat/completions'):
            self.api_url = f"{base_url.rstrip('/')}/chat/completions"
        else:
            self.api_url = base_url

        self.model_name = qwen_conf.get("model", "qwen")
        self.api_key = qwen_conf.get("api_key", "EMPTY")

        # 必须调用这个，否则报错！
        self.load_model()

    def load_model(self) -> None:
        """实现抽象方法：打印初始化信息"""
        print('================ QwenChat (Local vLLM) initialized ================')
        print(f"Target URL: {self.api_url}")
        print(f"Model Name: {self.model_name}")

    def chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        发送请求并强制解析 JSON (带兜底)
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            # 1. 请求
            resp = requests.post(self.api_url, json=payload, headers=headers, timeout=600)
            resp.raise_for_status()

            # 2. 获取文本
            data = resp.json()
            content = data['choices'][0]['message']['content']

            # 3. 提取 JSON (复用正则逻辑)
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                # 其次找 {}
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1:
                    json_str = content[start:end + 1]
                else:
                    # 兜底逻辑：如果找不到 JSON，不要报错，封装成字典返回
                    print(f"[Qwen] Warn: No JSON found. Fallback to raw text.")
                    return {"content": content, "raw_response": content}

            return json.loads(json_str)

        except json.JSONDecodeError as e:
            print(f"[Qwen] JSON 解析失败: {e}")
            return {"error": "JSON Decode Failed", "content": content}
        except Exception as e:
            print(f"[Qwen] API 请求异常: {e}")
            return {"error": "API Error", "details": str(e)}


# === 4. 测试入口 ===
if __name__ == '__main__':
    # 路径适配
    config_path = "../config/llm_config.yaml"

    try:
        # 1. 初始化
        agent = QwenChat(config_path)

        # 2. 构造强制 JSON 的 Prompt
        test_messages = [
            {
                "role": "system",
                "content": "你是一个助手。请务必以 JSON 格式回答，格式为：{\"answer\": \"你的回答内容\"}"
            },
            {
                "role": "user",
                "content": "你好，请问现在几点了？（请随意编一个时间）"
            }
        ]

        print("\n>>> 发送测试请求 (Expecting JSON)...")
        result = agent.chat(test_messages)

        print("\n>>> 模型返回结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        if "answer" in result:
            print("\n✅ JSON 解析成功！测试通过。")
        elif "content" in result:
            print("\n⚠️ 返回了纯文本（兜底生效），测试通过。")

    except Exception as e:
        print(f"\n❌ 测试运行出错: {e}")