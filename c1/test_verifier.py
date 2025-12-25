# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：test_verifier.py
@Author  ：niu
@Date    ：2025/12/25 11:32 
@Desc    ：
"""

# c1/test_verifier.py
import sys
import os

# 路径适配：确保能找到上级目录的包
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# 导入
from general.model import ZhipuChat  # 请确保您配置好了 llm_config.yaml
from general.decoupled_memory import DecoupledMemoryAtom
from c1.verifier import ConsistencyVerifier


def test_hallucination_removal():
    print(">>> 开始测试：基于真值过滤的语义剪枝 (Hallucination Removal) ...")

    # 1. 初始化模型
    config_path = os.path.join(root_dir, "config", "llm_config.yaml")
    try:
        llm = ZhipuChat(config_path)
    except Exception as e:
        print(f"模型加载失败，请检查配置文件: {e}")
        return

    verifier = ConsistencyVerifier(llm)

    # 2. 构造一个包含“幻觉”和“过度推断”的测试场景
    # 场景：原文只说了学Python，没提Java；只说了不喜欢C++，没说C++是垃圾。
    raw_text = "我最近在学 Python，觉得列表推导式很难。我不喜欢写 C++，语法太繁琐了。"

    # 模拟 Decoupler 提取出的脏数据 (Dirty Atoms)
    dirty_atoms = [
        # [1] 真实提取 (应该保留)
        DecoupledMemoryAtom(content="用户正在学习 Python，认为列表推导式有难度", atom_type="event"),

        # [2] 幻觉 (原文没提 Java -> 应该剪枝)
        DecoupledMemoryAtom(content="用户精通 Java 语言", atom_type="knowledge"),

        # [3] 真实提取 (应该保留)
        DecoupledMemoryAtom(content="用户不喜欢 C++，因为语法繁琐", atom_type="rule"),

        # [4] 过度推断 (原文只说繁琐，没说垃圾 -> 应该剪枝)
        DecoupledMemoryAtom(content="用户认为 C++ 是垃圾语言", atom_type="rule")
    ]

    print(f"\n[原始文本]: {raw_text}")
    print(f"[待检测原子]: {len(dirty_atoms)} 条")
    for a in dirty_atoms:
        print(f" - {a.content}")

    # 3. 执行校验
    print("\n>>> 正在运行 Verifier 生成探针...")
    clean_atoms = verifier.verify_batch(dirty_atoms, raw_text)

    # 4. 结果验证
    print(f"\n>>> 校验完成！剩余 {len(clean_atoms)} 条有效原子:")
    for atom in clean_atoms:
        print(f" [保留] {atom.content}")
        # 打印 LLM 的推理过程（Case Study 素材）
        if hasattr(atom, 'meta_data') and 'verification_log' in atom.meta_data:
            log = atom.meta_data['verification_log']
            print(f"      └-> 探针: {log['probe']}")
            print(f"      └-> 验证: {log['reasoning']}")

    # 5. 简单断言
    contents = [a.content for a in clean_atoms]
    has_java = any("Java" in c for c in contents)
    has_trash = any("垃圾" in c for c in contents)
    has_python = any("Python" in c for c in contents)

    if not has_java and not has_trash and has_python:
        print("\n✅ 测试通过！成功剔除了 'Java' 和 '垃圾语言' 等幻觉，实现了语义降噪。")
    else:
        print("\n❌ 测试失败！幻觉未被完全剔除。")


if __name__ == "__main__":
    test_hallucination_removal()
