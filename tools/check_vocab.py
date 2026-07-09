import json

vocab = json.load(open('checkpoints/vocab.json', encoding='utf-8'))
print(f'词汇表大小: {len(vocab["word2idx"])}')
print(f'部分词汇: {list(vocab["word2idx"].keys())[:30]}')
print(f'\n检查<unk>是否在词汇表中: {"<unk>" in vocab["word2idx"]}')
print(f'检查<eos>是否在词汇表中: {"<eos>" in vocab["word2idx"]}')
