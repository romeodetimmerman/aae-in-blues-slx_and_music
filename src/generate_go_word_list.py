import pandas as pd


def generate_go_word_list():
    """
    function to create go word list from entire corpus to use for MAXQDA keyword-in-context search
    """
    # set filename variable
    filename_codes = "corpus_studio.csv"

    # import raw csv files
    df_codes_raw = pd.read_csv(f"../data/raw/{filename_codes}")

    # initialize go word list
    go_words = df_codes_raw["Segment"].unique()

    # create go words file
    with open("../data/interim/go_words.txt", "w") as fp:
        for item in go_words:
            # write each item on a new line
            fp.write("%s\n" % item)
        print("Done")
