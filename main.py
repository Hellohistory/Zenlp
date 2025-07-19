from zenlp import discover
from pprint import pprint

# 准备一份你的语料（可以是任何字符串的可迭代对象）
corpus = [
    "大语言模型正在引领新一轮的技术革命。",
    "生成式AI的快速发展对内容创作产生了深远影响。",
    "遥遥领先的技术优势使得这家公司备受瞩目。",
    "赛博朋克风格的艺术作品充满了对未来的想象。",
    "这家公司的遥遥领先，得益于其强大的自研芯片。",
    "许多人对生成式AI的未来既期待又担忧。",
    "赛博朋克不仅仅是一种美学，更是一种文化现象。",
]

# 调用 discover 函数
# min_freq: 词语出现的最小频率
# min_pmi: 最小凝聚度 (内部关联性)
# min_entropy: 最小自由度 (外部多样性)
new_words = discover(
    corpus_source=corpus,
    min_freq=2,
    min_pmi=1.0,
    min_entropy=0.5
)

# 打印发现的新词及其分数
pprint(new_words)

