import pandas as pd
import multiprocessing as mp
import os
import utils
from aggregate_account_scores import aggregate_accounts_scores

from frameAxis import FrameAxis

lock = mp.Lock()
dirname = os.path.dirname(__file__)

# multiprocessing parameters
NUMBER_OF_PROCESSES = 1
CHUNK_SIZE = 100000

# scoring parameters
DICT_TYPE = 'emfd'
COL_NAME = "tweet_text"
# path to the word embedding model
WORD_EMEDDING_MODEL = "gensim-data\\word2vec-google-news-300\\word2vec-google-news-300\\GoogleNews-vectors-negative300.bin"

# setup the model
model = utils.setup_model(WORD_EMEDDING_MODEL)

# paths and file names
# rembember to add the \{}.csv at the end
OUT_CSV_PATH = "data\\scores\\frame-axis\\raw\\{}-frame-axis.csv"
IN_CSV_PATH = "data\\tweets\\{}.csv"

# names of file inside IN_CSV_PATH folder
FILE_NAMES = ["right-regular{}", "left-regular{}"]

# headers
HEADERS = ["user_id", "bias_fairness", "bias_care", "bias_loyalty", "bias_authority", "bias_sanctity",
           "intensity_fairness", "intensity_care", "intensity_loyalty", "intensity_authority", "intensity_sanctity",
           "fairness.virtue", "fairness.vice", "care.virtue", "care.vice", "loyalty.virtue", "loyalty.vice",
           "authority.virtue", "authority.vice", "sanctity.virtue", "sanctity.vice"]

AGGREGATED_OUTPUT_PATH = "data\\scores\\frame-axis\\aggregated"

KEYWORD = "#climatechange"


def filter_tweets(tweets_df, keyword):
    filtered_df = tweets_df[tweets_df["tweet_text"].str.contains(
        keyword, case=False)]

    return filtered_df


def calculate_score(data, output_path):
    print("Running FrameAxis Moral Foundations scores")

    if DICT_TYPE not in ["emfd", "mfd", "mfd2", "customized"]:
        raise ValueError(
            f'Invalid dictionary type received: {DICT_TYPE}, dict_type must be one of \"emfd\", \"mfd\", \"mfd2\", \"customized\"')

    fa = FrameAxis(mfd=DICT_TYPE, w2v_model=model)

    with lock:
        mf_scores = fa.get_fa_scores(df=data, doc_colname=COL_NAME, tfidf=False, format="virtue_vice",
                                     save_path=output_path)


def main():
    df_chunks = {}

    # read every file
    for name in FILE_NAMES:
        print(f"Reading {name}")

        # set paths
        input_path = IN_CSV_PATH.format(name.format(""))
        output_path = OUT_CSV_PATH.format(name.format(KEYWORD))

        # read csv
        df = pd.read_csv(input_path, on_bad_lines='skip', encoding='utf-8')
        df = df.astype("string")
        df = filter_tweets(df, KEYWORD)

        # split data
        df_chunks[name] = [df[i:i+CHUNK_SIZE].reset_index()
                           for i in range(0, len(df), CHUNK_SIZE)]

        # create NUMBER_OF_PROCESSES subchunks
        chunks = df_chunks[name]
        subchunks = [chunks[i:i+NUMBER_OF_PROCESSES]
                     for i in range(0, len(chunks), NUMBER_OF_PROCESSES)]

        # write file header
        output_file = open(output_path, "a", encoding="utf-8")
        utils.write_file(output_file, *HEADERS)
        output_file.close()

        # run processes
        for chunks in subchunks:
            processes = [mp.Process(target=calculate_score, args=(
                chunk, output_path)) for chunk in chunks]

            for p in processes:
                p.start()

            for p in processes:
                p.join()

        print("Aggregating")

        # aggregate users scores means
        file_to_aggregate_path = OUT_CSV_PATH.format(name.format(KEYWORD))

        to_agg_df = pd.read_csv(file_to_aggregate_path)
        print(to_agg_df.head())

        aggregated_file_path = f"{AGGREGATED_OUTPUT_PATH}\\{name.format(KEYWORD)}-aggregated.csv"
        aggregate_accounts_scores(file_to_aggregate_path, aggregated_file_path)


if __name__ == "__main__":
    main()
