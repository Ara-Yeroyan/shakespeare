import os
import requests
from loguru import logger

enhanced_path = 'data/enhanced'
os.makedirs(enhanced_path, exist_ok=True)

file_name = "shakespeare.txt"
if not os.path.isfile(file_name):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data = requests.get(url)

    with open(os.path.join(enhanced_path, file_name), 'w') as f:
        f.write(data.text)
    logger.info('Succesfully downloaded the data')

with open(os.path.join(enhanced_path, file_name), 'r') as f:
    texts = f.readlines()

test_portion = 0.1
idx = int(len(texts) * (1 - test_portion))
while not texts[idx] == '\n':  # test data should start from a valid block (not in the middle of a conversation
    idx += 1

test_texts = texts[idx:]
train_texts = texts[:idx]
logger.debug(f"Number of Train samples: {len(train_texts)}, Number of Test samples: {len(test_texts)}")

with open(os.path.join(enhanced_path, 'test.txt'), 'w') as f:
    f.write(''.join(test_texts))

with open(os.path.join(enhanced_path, 'train.txt'), 'w') as f:
    f.write(''.join(train_texts))
logger.info('Succesfully processed and splitted the data!')
