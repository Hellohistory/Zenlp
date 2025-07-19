# tests/discovery/test_discoverer.py
import pytest
from zenlp import discover
import os

CORPUS_CONTENT = (
    "机器学习是未来的方向，深度学习也是机器学习的一个分支。\n"
    "自然语言处理也很有趣，自然语言处理是AI的重要领域。"
)
CORPUS_AS_LIST = CORPUS_CONTENT.strip().split('\n')


@pytest.fixture
def temp_corpus_path():
    temp_path = "temp_test_corpus_for_discovery.txt"
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(CORPUS_CONTENT)
    yield temp_path
    os.remove(temp_path)


# 【修改点 1】: 将 lazy_fixture 调用改为传递 fixture 的名字（字符串）
@pytest.mark.parametrize(
    "corpus_input",
    [
        "temp_corpus_path",  # <--- 修改这里
        CORPUS_AS_LIST
    ],
    ids=["from_file_path", "from_memory_list"]
)
# 【修改点 2】: 在测试函数签名中加入 pytest 内置的 request fixture
def test_discover_basic_functionality(corpus_input, request):
    """
    测试：验证核心功能（现在能同时处理文件路径和内存列表）。
    """
    # 【修改点 3】: 如果输入是字符串（即fixture的名字），通过 request 解析它
    if isinstance(corpus_input, str):
        corpus_input = request.getfixturevalue(corpus_input)

    found_words = discover(
        corpus_source=corpus_input,
        max_word_len=6,
        min_freq=2,
        min_pmi=1.0,
        min_entropy=0.5
    )
    assert "机器学习" in found_words
    assert "自然语言处理" in found_words
    assert found_words["机器学习"]['freq'] == 2
    assert found_words["自然语言处理"]['freq'] == 2


def test_discover_with_output_file(temp_corpus_path): # 此测试仍依赖文件路径
    """
    测试：验证文件输出功能。
    """
    output_path = "temp_new_words_output.txt"
    # 修改：调用 discover 时使用新的参数名 corpus_source
    found_words = discover(
        corpus_source=temp_corpus_path,
        output_path=output_path,
        max_word_len=6,
        min_freq=2,
        min_pmi=1.0,
        min_entropy=0.5
    )

    assert os.path.exists(output_path), "未能成功创建输出文件"
    assert len(found_words) > 0, "在放宽所有阈值后，依然没有发现任何新词"

    with open(output_path, 'r', encoding='utf-8') as f:
        content = f.read()
        assert "word\tfreq\tpmi" in content, "输出文件中缺少表头"
        assert "机器学习" in content, "输出文件中缺少关键词'机器学习'"
        assert "自然语言处理" in content, "输出文件中缺少关键词'自然语言处理'"

    os.remove(output_path)


def test_discover_empty_corpus():
    """测试：处理空语料库的边界情况（空文件和空列表）。"""
    # 场景1：空文件
    empty_corpus_path = "empty_corpus.txt"
    with open(empty_corpus_path, "w") as f: f.write("")
    found_words_from_file = discover(corpus_source=empty_corpus_path)
    assert found_words_from_file == {}
    os.remove(empty_corpus_path)

    # 场景2：空列表
    found_words_from_list = discover(corpus_source=[])
    assert found_words_from_list == {}


def test_discover_file_not_found():
    """测试：当文件不存在时，能否正确抛出FileNotFoundError异常。"""
    with pytest.raises(FileNotFoundError):
        # 修改：调用 discover 时使用新的参数名 corpus_source
        discover(corpus_source="a_file_that_absolutely_does_not_exist.txt")