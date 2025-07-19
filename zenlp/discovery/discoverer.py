# zenlp/discovery/discoverer.py

"""ZenLP 的新词发现模块。

本模块提供了基于统计指标（如点互信息PMI和边界熵）
从原始文本语料中发现新词的功能。
"""
import logging
import math
import os
from collections import defaultdict, Counter
from typing import Iterable, List, Union, Dict, Any

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='[ZenLP Discovery] %(message)s')


class _NgramStats:
    """用于存储所有N-gram相关统计信息的内部数据容器。"""

    def __init__(self):
        """初始化统计容器。"""
        self.freq_map = Counter()
        self.left_neighbors = defaultdict(Counter)
        self.right_neighbors = defaultdict(Counter)
        self.total_chars = 0

    def scan(self, corpus_source: Union[str, List[str], Iterable[str]], max_word_len: int):
        """扫描语料源以收集N-gram统计信息。

        参数:
            corpus_source (Union[str, List[str], Iterable[str]]): 语料的来源。
                可以是一个单独的文件路径（str），一个文件路径的列表（list），
                或者是一个字符串的可迭代对象（如列表）。
            max_word_len (int): 需要考虑的N-gram的最大长度。

        引发:
            FileNotFoundError: 如果提供的文件路径不存在。
            TypeError: 如果 corpus_source 的类型不被支持。
        """
        is_file_path_mode = False
        if isinstance(corpus_source, str):
            if not os.path.isfile(corpus_source):
                raise FileNotFoundError(f"没有该文件或目录: '{corpus_source}'")
            corpus_source = [corpus_source]
            is_file_path_mode = True

        logging.info("步骤 1/3: 开始扫描语料库...")

        if is_file_path_mode or (isinstance(corpus_source, list) and all(
                isinstance(p, str) and os.path.isfile(p) for p in corpus_source)):
            lines = self._iter_files(corpus_source)
            try:
                total_size = sum(os.path.getsize(p) for p in corpus_source)
                pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="  -> 扫描文件")
                for line, byte_inc in lines:
                    pbar.update(byte_inc)
                    self._process_line(line, max_word_len)
                pbar.close()
            except Exception:  # 降级处理无法获取大小的迭代器
                for line, byte_inc in lines:
                    self._process_line(line, max_word_len)
        elif isinstance(corpus_source, Iterable):
            for line in tqdm(corpus_source, desc="  -> 扫描迭代器"):
                self._process_line(line, max_word_len)
        else:
            raise TypeError("corpus_source 必须是文件路径字符串或字符串的可迭代对象。")

    def _iter_files(self, paths: List[str]):
        """一个生成器，逐行读取文件列表以节省内存。"""
        for path in paths:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    byte_inc = len(line.encode('utf-8'))
                    yield line.rstrip('\n'), byte_inc

    def _process_line(self, line: str, max_word_len: int):
        """处理单行文本以更新统计数据。"""
        if not line:
            return
        self.total_chars += len(line)
        for i in range(len(line)):
            left_char = line[i - 1] if i > 0 else '<s>'
            for k in range(1, max_word_len + 1):
                if i + k > len(line):
                    break
                word = line[i:i + k]
                self.freq_map[word] += 1
                if len(word) > 1:
                    self.left_neighbors[word][left_char] += 1
                    right_char = line[i + k] if i + k < len(line) else '</s>'
                    self.right_neighbors[word][right_char] += 1


def _calculate_scores(stats: _NgramStats, min_freq: int, min_pmi: float, min_entropy: float) -> Dict[
    str, Dict[str, Any]]:
    """计算PMI和熵分数，并筛选出新词。"""

    def _entropy(counts: Counter) -> float:
        """计算香农熵的辅助函数。"""
        total = sum(counts.values())
        if total <= 1:
            return 0.0
        return sum(-(c / total) * math.log2(c / total) for c in counts.values())

    logging.info("步骤 2/3: 开始计算凝聚度(PMI)和自由度(Entropy)...")

    potential_words = {word: freq for word, freq in stats.freq_map.items()
                       if freq >= min_freq and len(word) > 1}
    new_words = {}

    pbar = tqdm(total=len(potential_words), desc="  -> 计算分数")
    for word, freq in potential_words.items():
        pbar.update(1)
        # 拉普拉斯平滑 (加一) 以避免零概率。
        p_word = (freq + 1) / (stats.total_chars + 1)

        pmi_score = float('inf')
        # “木桶效应”原则：一个词的PMI取决于其所有二元切分中最小的那个。
        for i in range(1, len(word)):
            prefix, suffix = word[:i], word[i:]
            p_p1 = (stats.freq_map.get(prefix, 0) + 1) / (stats.total_chars + 1)
            p_p2 = (stats.freq_map.get(suffix, 0) + 1) / (stats.total_chars + 1)
            current_pmi = math.log2(p_word / (p_p1 * p_p2))
            pmi_score = min(pmi_score, current_pmi)

        if pmi_score < min_pmi:
            continue

        left_entropy = _entropy(stats.left_neighbors[word])
        right_entropy = _entropy(stats.right_neighbors[word])
        combined_entropy = min(left_entropy, right_entropy)

        if combined_entropy < min_entropy:
            continue

        new_words[word] = {
            'freq': freq,
            'pmi': round(pmi_score, 2),
            'left_entropy': round(left_entropy, 2),
            'right_entropy': round(right_entropy, 2)
        }
    pbar.close()
    return new_words


def discover(
        corpus_source: Union[str, Iterable[str]],
        output_path: str = None,
        max_word_len: int = 5,
        min_freq: int = 10,
        min_pmi: float = 1.5,
        min_entropy: float = 1.5
) -> Dict[str, Dict[str, Any]]:
    """使用统计指标从大型语料库中发现新词。

    这是主公开API，它协调了从扫描语料到计算分数、筛选结果的整个新词发现流程。

    参数:
        corpus_source (Union[str, Iterable[str]]): 语料的来源。
            可以是一个单独的文件路径 (str) 或一个字符串的可迭代对象 (如列表)。
        output_path (str, optional): 保存已发现新词的文件路径。
            如果为 None，则不保存结果到文件。默认为 None。
        max_word_len (int, optional): 考虑的词的最大长度。
            默认为 5。
        min_freq (int, optional): 一个词被视为候选词的最低频率。
            默认为 10。
        min_pmi (float, optional): 一个词被认为具有凝聚力的最低点互信息 (PMI) 分数。
            默认为 1.5。
        min_entropy (float, optional): 一个词的边界被认为是自由的最低边界熵。
            默认为 1.5。

    返回:
        Dict[str, Dict[str, Any]]: 一个字典，包含了发现的词及其相关的统计数据
        （'freq', 'pmi', 'left_entropy', 'right_entropy'）。
    """
    stats_container = _NgramStats()
    stats_container.scan(corpus_source, max_word_len)
    found_words = _calculate_scores(stats_container, min_freq, min_pmi, min_entropy)

    logging.info(f"步骤 3/3: 完成！发现 {len(found_words)} 个潜在新词。")

    if output_path:
        logging.info(f"正在将结果保存到 {output_path}...")
        try:
            # 在保存前按频率降序排序。
            sorted_words = sorted(found_words.items(), key=lambda item: item[1]['freq'], reverse=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("word\tfreq\tpmi\tleft_entropy\tright_entropy\n")
                for word, stats in sorted_words:
                    f.write(
                        f"{word}\t{stats['freq']}\t{stats['pmi']}\t{stats['left_entropy']}\t{stats['right_entropy']}\n")
            logging.info("保存完成。")
        except Exception as e:
            logging.error(f"保存结果时发生错误: {e}")

    # 注意：函数返回的是发现时原始的字典，而非排序后的。
    # 如果需要排序后的结果，请直接使用 `sorted_words`。
    return found_words