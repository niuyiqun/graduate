# utils/add_embedding_patch.py
import json
import os
import sys
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# === 配置 ===
INPUT_FILE = "c1/output/locomo_extracted_atoms_no_embedding.jsonl"
OUTPUT_FILE = "c1/output/locomo_extracted_atoms_with_emb.jsonl"
MODEL_PATH = "model/all-MiniLM-L6-v2"  # 你的本地模型路径


def add_embeddings():
    # 1. 加载模型
    print(f"正在加载 Embedding 模型: {MODEL_PATH} ...")
    if os.path.exists(MODEL_PATH):
        model = SentenceTransformer(MODEL_PATH)
    else:
        print("本地模型不存在，正在下载 all-MiniLM-L6-v2 ...")
        model = SentenceTransformer('all-MiniLM-L6-v2')

    # 2. 读取数据
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return

    print(f"正在处理数据: {INPUT_FILE}")

    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
            open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:

        lines = fin.readlines()

        for line in tqdm(lines, desc="Processing Samples"):
            if not line.strip(): continue

            try:
                sample = json.loads(line)

                # 获取 memory_atoms 列表
                atoms = sample.get("memory_atoms", [])

                # 收集需要计算的文本
                texts = [atom.get("content", "") for atom in atoms]

                # 批量计算 Embedding (Batch Inference) -> 速度极快
                if texts:
                    embeddings = model.encode(texts, show_progress_bar=False)

                    # 将 Embedding 回写到 atom 对象中
                    for i, atom in enumerate(atoms):
                        # 转为 list 以便 JSON 序列化
                        atom["embedding"] = embeddings[i].tolist()

                # 写入新文件
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"Skipping bad line: {e}")

    print(f"✅ 处理完成！")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"请修改 c2/pipeline.py 中的 C1_OUTPUT 路径指向新文件！")


if __name__ == "__main__":
    # 确保脚本在项目根目录下也能运行
    sys.path.append(os.getcwd())
    add_embeddings()