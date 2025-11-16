import torch

# 加载 .pt 文件（只查看结构，不执行复杂解析）
model_path = "emnlp2022_syngec_biaffine-dep-electra-zh-gopar.pt"
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

# 打印顶层键和每个键的值类型/长度
print("文件顶层键：", list(checkpoint.keys()))
for key in checkpoint.keys():
    value = checkpoint[key]
    print(f"\n键 '{key}' 的信息：")
    print(f"  - 类型：{type(value)}")
    if isinstance(value, (list, tuple)):
        print(f"  - 长度：{len(value)}")  # 重点看是否有长度超过 4 的序列
    elif isinstance(value, dict):
        print(f"  - 包含键数：{len(value.keys())}")

# print args
if 'args' in checkpoint:
    args = checkpoint['args']
    print("\n模型参数（args）：")
    for arg_key, arg_value in vars(args).items():
        print(f"  - {arg_key}: {arg_value}")