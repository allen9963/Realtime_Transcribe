from opencc import OpenCC

converter = OpenCC("s2t")
print(converter.convert("简体中文测试"))
