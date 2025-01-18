### Read text from file ###
import urllib.request
import re

def getVocabulary():
    url = ("https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_words = sorted(set(preprocessed))
    vocab = {token:integer for integer,token in enumerate(all_words)}

    return vocab, preprocessed, raw_text
if __name__ == "__main__":
    vocab, preprocessed, raw_text = getVocabulary()
    print("Total number of character:", len(raw_text))
    print(raw_text[:99])

    ### Simple Tokenization by splitting line of text into parts###
    text = "Hello, world. This, is a test."
    result = re.split(r'(\s)', text)
    print(result)

    result = re.split(r'([,.]|\s)', text)
    print(result)

    result = [item for item in result if item.strip()]
    print(result)

    text = "Hello, world. Is this-- a test?"
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    result = [item.strip() for item in result if item.strip()]
    print(result)

    # If you noticed above, more character we specify as delimeter in split function, 
    # more capable it becomes to split the text into tokens.

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(len(preprocessed))
    print(preprocessed[:30])

    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    print(vocab_size)

    vocab = {token:integer for integer,token in enumerate(all_words)}
    for i, item in enumerate(vocab.items()):
        print(item)
        if i >= 50:
            break
