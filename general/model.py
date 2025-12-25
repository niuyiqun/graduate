# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：model.py
@Author  ：niu
@Date    ：2025/12/5 09:02 
@Desc    ：
"""
from abc import ABC, abstractmethod

import yaml
import json
from abc import abstractmethod, ABC
from typing import List, Dict, Any

import requests
import torch
import transformers
import yaml
from transformers import AutoTokenizer
from zhipuai import ZhipuAI

class BaseModel(ABC):
    def __init__(self, config_path: str) -> None:
        """
        初始化基类，加载配置文件
        :param config_path: 配置文件路径
        """
        self.config = self.load_config(config_path)

    def load_config(self, config_path: str = './config/llm_config.yaml') -> Dict[str, Dict]:
        """
        从配置文件加载 YAML 配置
        :param config_path: 配置文件路径
        :return: 配置字典
        """
        try:
            with open(config_path, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    @abstractmethod
    def chat(self, prompt: str) -> str:
        """
        抽象方法，用于与模型交互。
        子类需要实现此方法。
        """
        pass

    @abstractmethod
    def load_model(self) -> None:
        """
        抽象方法，用于加载模型。
        子类需要实现此方法。
        """
        pass


'''非pipeline形式'''


# from transformers import AutoModelForCausalLM
# from transformers import AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained("./weight/Llama-2-7b-chat-hf",
#                                           padding_side="left")
# tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained('./weight/Llama-2-7b-chat-hf',
#                                              device_map="auto",
#                                              load_in_8bit=True
#                                              )
# last_five_actions: List[str] = []
# last_five_states: List[str] = []
#
# message = [
#     {
#         "role": "system",
#         "content": "You're a helpful game expert, Your task is to play a game based on natural language text."
#                    "Please first think step-by-step and then tell what to do next. "
#                    "Organize your output in a json formatted as "
#                    "Dict{'step_by_step_thinking': Str(explanation),'action': Str{action}}. "
#     },
#     {
#         "role": "user",
#         "content": "",
#         "last five action": 1
#     }
# ]
#
# model_inputs = tokenizer(['Explain antibiotics '
#                           'A:'
#                           'Please do not repeat the question, give a direct answer'], return_tensors="pt").to("cuda")
# generated_ids = model.generate(**model_inputs,
#                                max_new_tokens=50,
#                                do_sample=True,
#                                temperature=0.5,
#                                )
# ans = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
# print(ans)

class LlamaChat(BaseModel):
    def __init__(self, config_path: str = './config/llm_config.yaml') -> None:
        super().__init__(config_path)
        llama_config = self.config.get("llama2", {})
        self.api_url = llama_config.get("api_url", "http://localhost:8000/generate")
        if not self.api_url:
            raise ValueError("API URL is not provided in the configuration.")
        self.load_model()

    def chat(self, prompt: List[Dict[str, Any]]) -> Dict[str, str]:
        # Define the payload for the API request
        payload = {
            "prompt": json.dumps(prompt),
            "temperature": 0.7,
            "top_k": 10,
            "max_tokens": 150,
            "stop": None  # Add stop tokens if necessary
        }
        try:
            # Send POST request to the API
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
            result = response.json()

            # Return the generated text
            return {
                "prompt": prompt,
                "response": result.get("text", "No response generated")
            }

        except requests.exceptions.RequestException as e:
            # Handle errors in the request
            raise RuntimeError(f"Error during API request: {e}")

        # if self.pipeline is None:
        #     raise RuntimeError("Model pipeline is not loaded.")
        # answer = self.pipeline(
        #     prompt,
        #     do_sample=True,
        #     top_k=10,
        #     eos_token_id=self.tokenizer.eos_token_id,
        #     max_new_tokens=150,
        #     truncation=True
        # )[0]
        # return answer

    def load_model(self) -> None:
        """
                This function is a placeholder since the model is no longer loaded locally.
        """
        print('================ Model API ready ================')
        print("LlamaChat initialized with API URL:", self.api_url)


class ZhipuChat(BaseModel):
    def __init__(self, config_path: str = './config/llm_config.yaml') -> None:
        """
        初始化 ZhipuChat
        :param config_path: 配置文件路径
        """
        super().__init__(config_path)
        zhipu_config = self.config.get("zhipu", {})
        self.api_key = zhipu_config.get("api_key", "")
        self.model = zhipu_config.get("model", "glm-4")
        self.client = None
        self.load_model()

    def load_model(self) -> None:
        """
        加载 ZhipuAI 客户端
        """
        print('================ Loading ZhipuAI client ================')
        self.client = ZhipuAI(api_key=self.api_key)
        print('================ ZhipuAI client loaded ================')

    def chat(self, messages: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        与 ZhipuAI 模型交互 (增强版：正则提取，专治各种格式不服)
        """
        if self.client is None:
            raise RuntimeError("ZhipuAI client is not loaded.")

        try:
            # 1. 调用 API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )

            # 获取原始文本
            content = response.choices[0].message.content

            # 2. 【核心修复】使用正则表达式提取 JSON
            import re

            # 解释：
            # ```(?:json)?  -> 匹配 ``` 开头，json 可有可无
            # \s* -> 忽略中间的换行符或空格
            # (\{.*?\})     -> 【核心捕获】抓取 { ... } 中间的所有内容
            # \s* -> 忽略后面的空白
            # ```           -> 匹配结尾的 ```
            # re.DOTALL     -> 让 . 也能匹配换行符，关键！
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)

            if json_match:
                # 方案 A：如果找到了 Markdown 框，直接取框里的
                json_str = json_match.group(1)
            else:
                # 方案 B：如果没框，尝试暴力寻找最外层的 {}
                # 这能处理开头有一堆废话的情况，比如 "好的，这是您的JSON：\n{...}"
                start_idx = content.find("{")
                end_idx = content.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    json_str = content[start_idx: end_idx + 1]
                else:
                    # 方案 C：实在没招了，硬解析原始文本
                    json_str = content

            # 3. 解析 JSON
            parsed_answer = json.loads(json_str)
            return parsed_answer

        except json.JSONDecodeError as json_error:
            print(f"JSON parsing error: {json_error}")
            print(f"Failed JSON content: {content}")  # 打印错误现场
            return {"error": "JSON parsing error", "details": str(json_error)}

        except Exception as e:
            print(f"Error during chat: {e}")
            return {"error": "Other error", "details": str(e)}


if __name__ == '__main__':
    config_path = "../config/llm_config.yaml"
    agent = ZhipuChat(config_path)
    print('generated content:', agent.chat('messages'))