import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Chia câu thành một mảng chữ/kí tự
    kí tự có thể là chữ cái, dấu câu hay chữ số
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = tìm từ gốc của một từ
    ví dụ
    words = ["contribution", "contributes", "contributing"]
    -> ["contribute", "contribute", "contribue"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    Trả lại một túi từ
    Gán giá trị 1 cho mỗi từ đã biết ở trong câu, và 0 cho các từ chưa biết
    Ví dụ
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag