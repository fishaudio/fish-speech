import os
import pickle
import re
from functools import lru_cache

from g2p_en import G2p

current_file_path = os.path.dirname(__file__)
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")

_g2p = G2p()


def read_dict():
    g2p_dict = {}
    start_line = 49
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= start_line:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0]

                syllable_split = word_split[1].split(" - ")
                g2p_dict[word] = []
                for syllable in syllable_split:
                    phone_split = syllable.split(" ")
                    g2p_dict[word].append(phone_split)

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def cache_dict(g2p_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)


@lru_cache(maxsize=1)
def get_dict():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict()
        cache_dict(g2p_dict, CACHE_PATH)

    return g2p_dict


def g2p(text):
    eng_dict = get_dict()

    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)

    for w in words:
        if w.upper() in eng_dict:
            phns = eng_dict[w.upper()]
            for ph in phns:
                phones += ph
            continue

        phones.extend(list(filter(lambda p: p != " ", _g2p(w))))

    return phones


if __name__ == "__main__":
    print(g2p("Hugging face, BGM"))
