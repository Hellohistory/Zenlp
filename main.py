from zenlp import discover

# 新词发掘调用演示
new_words = discover(
    corpus_path='new_word_test.txt',
    output_path='new_words.tsv',
    min_freq=20,
    min_pmi=1.5,
    min_entropy=1.5
)

print(f"Discovered {len(new_words)} new words.")
