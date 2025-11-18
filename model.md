# 模型改进
1. encoder：
+ 将syntax encoder和sentence encoder输出分离，原来的处理是加权求和，现在encoder输出两个独立向量

2. decoder：
+ 使用fmoe，将原来的ffn层替换为moe层，初始化expert权重使用baseline的ffn权重
+ layer0的gate使用syntax encoder输出，为了匹配维度，进行了一次cross-attention；，expert使用sentence encoder输出。
    + 具体见fmoe.transformer.SyntaxGuidedFMoETransformerMLP 
+ layer1-11的gate和expert均使用和sentence encoder输出。

baseline:
=========== Span-Based Correction ============
TP	FP	FN	Prec	Rec	F0.5
4369	4032	11617	0.5201	0.2733	0.4405
==============================================

# V1.0
1. 模型设置
+ gate: Switch topk=2
2. 训练信息
+ epoch4: loss ~3.0 cosine
+ decoder-only: 只训练了decoder部分，encoder部分参数冻结
3. result: 
=========== Span-Based Correction ============
TP	FP	FN	Prec	Rec	F0.5
3522	4656	12375	0.4307	0.2216	0.3623
==============================================
