# ZenLP - NLP工具包

![PyPI](https://img.shields.io/pypi/v/zenlp?label=PyPI&logo=pypi&color=blue)
![Python Versions](https://img.shields.io/pypi/pyversions/zenlp?logo=python&label=Python)
![Build Status](https://img.shields.io/github/actions/workflow/status/Hellohistory/zenlp/publish.yml)
![License](https://img.shields.io/github/license/Hellohistory/zenlp)

**大道至简，返璞归真。**

`ZenLP` 是一个为Python设计的极简、优雅的自然语言处理（NLP）工具包。

在大语言模型（LLM）席卷业界的今天，我们常常被其复杂的结构和高昂的资源需求所困扰。
`ZenLP` 回归初心，专注于那些历久弥新、效果稳健且具有高度可解释性的经典NLP算法。
我们相信，优雅的工程实现能让这些经典算法在现代NLP工作流中焕发新的光彩。

本项目旨在为每个算法提供：
* 清晰、完备的文档和注释。
* 符合直觉、易于使用的API。
* 生产就绪的性能和稳定性。

欢迎来到ZenNLP的世界，在这里，我们用最少的代码，解决最核心的问题。

## ✨ 特性

* **极简设计**: 摒弃繁杂的依赖和配置，每个模块都力求小而美，专注做好一件事。
* **无监督与自适应**: 核心算法多为无监督实现，能自动从原始文本中学习，轻松适应不同领域。
* **生产就绪**: 所有代码均经过严格测试，并提供类型提示，确保在生产环境中的稳定性和可维护性。
* **教育友好**: 不仅是工具，更是学习经典NLP算法的优秀参考资料。每一行代码、每一篇文档都在传递算法的精髓。

## 🚀 快速开始

### 1. 安装

通过 pip 可以轻松安装 `ZenLP`：
```bash
pip install zenlp
```

如果因为配置镜像站无法获取，请强制使用官方PyPi获取
```bash
pip install zenlp -i https://pypi.org/simple --trusted-host pypi.org
```

### 2. 使用示例：发现新词
只需几行代码，即可从您的语料中发现新词。

```python
from zenlp import discover
from pprint import pprint

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

pprint(new_words)

```

## 📖 功能模块

`ZenLP` 将会逐步涵盖NLP中的多个核心领域，每个模块都遵循“禅”的设计哲学。

* ✅ **`zenlp.discovery` - 新词发现**
    * **功能**: 基于 `PMI + 左右熵` 的无监督新词发现。
    * **状态**: 已完成。
    * **技术原理**: [新词发掘](docs/new_word_discoverer.md)


## 🤝 贡献 (Contributing)

我们热烈欢迎任何形式的贡献！无论您是想修复一个Bug、增加一个新功能，还是改进文档，都请不要犹豫。

请在提交前确保您的代码通过了测试，并遵循了项目的编码风格。

## 📜 许可证 (License)

本项目采用 [MIT License](https://github.com/Hellohistory/zenlp/License) 授权。

---
Made with ❤️ and Zenlp.