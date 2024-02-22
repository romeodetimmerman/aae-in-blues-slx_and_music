import pandas as pd
import numpy as np
import wordninja


def fix_word_context():
    """
    function to use vocab list ordered by prevalence in corpus to parse words
    in "Next word" and "Previous word" columns of dataframe
    """
    wordninja.DEFAULT_LANGUAGE_MODEL = wordninja.LanguageModel(
        "../data/interim/vocab_list.txt.gz"
    )

    df = pd.read_csv("../data/interim/corpus_studio_pre_processed.csv")

    indices_to_drop = []

    for index, row in enumerate(df["Next word"]):
        if isinstance(row, float) and np.isnan(row):  # NaN values
            df.at[index, "Next word"] = ""  # replace with empty string
        else:
            words = wordninja.split(row)
            if len(words) < 3:
                df.at[index, "Next word"] = words[0]
            else:
                indices_to_drop.append(index)

    for index, row in enumerate(df["Previous word"]):
        if isinstance(row, float) and np.isnan(row):  # NaN values
            df.at[index, "Previous word"] = ""  # replace with empty string
        else:
            words = wordninja.split(row)
            if len(words) < 3:
                df.at[index, "Previous word"] = words[-1]
            else:
                indices_to_drop.append(index)

    # Drop rows with indices collected in indices_to_drop list
    df.drop(indices_to_drop, inplace=True)

    df.to_csv(
        "../data/interim/corpus_studio_pre_processed.csv", index=False, encoding="UTF-8"
    )
