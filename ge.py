# -*- coding: UTF-8 -*-

import os


def generate_simple_tree(startpath, max_depth=2):
    exclude_dirs = {'.git', '__pycache__', 'venv', '.vscode', '.idea'}

    # 获取规范化的起始路径
    startpath = os.path.abspath(startpath)
    prefix_length = len(startpath.rstrip(os.sep))

    for root, dirs, files in os.walk(startpath):
        # 计算当前深度 (根目录深度为 0)
        relative_path = root[prefix_length:].strip(os.sep)
        depth = 0 if not relative_path else relative_path.count(os.sep) + 1

        # 如果超过最大深度，停止向下探测
        if depth >= max_depth:
            dirs[:] = []  # 清空 dirs 列表，阻止 os.walk 继续深入
            continue

        # 过滤不需要的目录
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]

        # 打印当前目录
        indent = '    ' * depth
        if depth == 0:
            print(f"【项目根目录】: {os.path.basename(startpath)}")
        else:
            print(f"{indent}└── {os.path.basename(root)}/")

        # 打印当前层级的文件
        file_indent = '    ' * (depth + 1)
        for f in files:
            if not f.startswith('.'):
                print(f"{file_indent}├── {f}")


# 运行：限制为 2 级
generate_simple_tree('.', max_depth=2)