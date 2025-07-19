# zenlp/__init__.py

"""
ZenLP: 一个为 Python 设计的极简、优雅的自然语言处理工具包。

本包旨在为经典和现代的NLP算法提供清晰、文档完备且生产就绪的实现。

可用函数:
- discover: 使用点互信息(PMI)和边界熵从原始文本语料中发现新词。
"""

__version__ = "0.1.0"
__author__ = "你的名字"

from .discovery.discoverer import discover

__all__ = [
    'discover',
]
