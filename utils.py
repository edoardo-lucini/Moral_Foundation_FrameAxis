import csv
from gensim.models import KeyedVectors

def write_file(file, *args):
    writer = csv.writer(file)
    writer.writerow(args)

def setup_model(word_emdedding):
    if word_emdedding is not None:
        W2V_PATH = word_emdedding
        model = KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)
    else:
        print('Downloading word embedding model: word2vec-google-news-300')
        import gensim.downloader
        model = gensim.downloader.load('word2vec-google-news-300')
    
    return model