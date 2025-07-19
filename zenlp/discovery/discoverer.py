# zenlp/discovery/discoverer.py
"""
ZenLP - 新词发现模块
"""
import logging
import math
import os
from collections import defaultdict, Counter
from typing import Iterable, List

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='[ZenLP Discovery] %(message)s')


class _NgramStats:
    """数据容器: 负责扫描语料并存储所有N-gram相关的统计数据。"""

    def __init__(self):
        self.freq_map = Counter()
        self.left_neighbors = defaultdict(Counter)
        self.right_neighbors = defaultdict(Counter)
        self.total_chars = 0

    def scan(self, corpus_source, max_word_len: int):
        """
        扫描语料，支持传入文件路径字符串或可迭代字符串列表。
        """
        if isinstance(corpus_source, str):
            corpus_source = [corpus_source]

        logging.info("步骤 1/3: 开始扫描语料库...")
        if all(os.path.isfile(p) for p in corpus_source):
            # 文件列表
            lines = self._iter_files(corpus_source)
            total_size = sum(os.path.getsize(p) for p in corpus_source)
            pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="  -> 扫描进度")
            for line, byte_inc in lines:
                pbar.update(byte_inc)
                self._process_line(line, max_word_len)
            pbar.close()
        elif isinstance(corpus_source, Iterable):
            # 任意字符串迭代器
            lines = corpus_source
            for line in lines:
                self._process_line(line, max_word_len)
        else:
            raise TypeError("corpus_source 必须是文件路径字符串或字符串可迭代对象")

    def _iter_files(self, paths: List[str]):
        for path in paths:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    byte_inc = len(line.encode('utf-8'))
                    yield line.rstrip('\n'), byte_inc

    def _process_line(self, line: str, max_word_len: int):
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


def _calculate_scores(stats: _NgramStats, min_freq: int, min_pmi: float, min_entropy: float) -> dict:
    """算法函数: 接收统计数据，执行计算并返回结果。"""

    def _entropy(counts: Counter) -> float:
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return sum(-(c / total) * math.log2(c / total) for c in counts.values())

    logging.info("步骤 2/3: 开始计算凝聚度(PMI)和自由度(Entropy)...")

    potential_words = {word: freq for word, freq in stats.freq_map.items()
                       if freq >= min_freq and len(word) > 1}
    new_words = {}

    pbar = tqdm(total=len(potential_words), desc="  -> 计算分数")
    for word, freq in potential_words.items():
        pbar.update(1)
        p_word = (freq + 1) / (stats.total_chars + 1)  # 加一平滑

        pmi_score = float('inf')
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


def discover(corpus_path: str, output_path: str = None, max_word_len: int = 5, min_freq: int = 10,
             min_pmi: float = 1.5, min_entropy: float = 1.5) -> dict:
    """公开API: 从指定的大型语料库中发现新词。"""
    stats_container = _NgramStats()
    stats_container.scan(corpus_path, max_word_len)
    found_words = _calculate_scores(stats_container, min_freq, min_pmi, min_entropy)

    logging.info(f"步骤 3/3: 完成！发现 {len(found_words)} 个潜在新词。")

    if output_path:
        logging.info(f"正在将结果保存到 {output_path}...")
        try:
            sorted_words = sorted(found_words.items(), key=lambda item: item[1]['freq'], reverse=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("word\tfreq\tpmi\tleft_entropy\tright_entropy\n")
                for word, stats in sorted_words:
                    f.write(
                        f"{word}\t{stats['freq']}\t{stats['pmi']}\t{stats['left_entropy']}\t{stats['right_entropy']}\n")
            logging.info("保存完成。")
        except Exception as e:
            logging.error(f"保存结果时发生错误: {e}")

    return found_words