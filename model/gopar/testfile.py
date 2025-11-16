import torch
import os
from supar import Parser

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 定义要加载的文件名（就是那个没有后缀的大文件）
model_file = "emnlp2022_syngec_biaffine-dep-electra-zh-gopar.pt"

print(f"尝试加载文件: {model_file}")

try:
    # 尝试用 torch.load 加载，不指定 map_location 会自动加载到可用的设备
    # 如果只想在 CPU 上加载，可以加上 map_location="cpu"
    dep = Parser.load(model_file)
    # model_weights = torch.load(model_file, map_location="cuda")
    
    # print("✅ 成功加载！文件是 PyTorch 权重格式。")
    
    # # 打印一些信息来确认
    # if isinstance(model_weights, dict):
    #     print(f"文件中包含 {len(model_weights)} 个权重项。")
    #     print("前 5 个权重项的键名：")
    #     for i, key in enumerate(list(model_weights.keys())[:5]):
    #         print(f"  {i+1}. {key}")
    # else:
    #     print("文件内容不是一个字典，而是一个:", type(model_weights))

except Exception as e:
    print(f"❌ 加载失败！错误信息：{e}")