import pandas as pd


def replace_apostrophes(text):
    """
    function to fix curly apostrophes
    """
    if isinstance(text, str):
        text = text.replace("’", "'")
        text = text.replace("‘", "'")
    return text


def make_data(df_codes_raw, df_context_raw, df_year_raw):
    """
    function to make processed data set after keyword-in-context search has been performed in MAXQDA
    """

    # prepare raw csv files for merge
    df_codes_raw.drop(
        columns=[
            "Comment",
            "End",
            "Modified by",
            "Created by",
            "Color",
            "Weight score",
            "Modified",
            "Area",
            "Coverage %",
        ],
        inplace=True,
    )

    df_context_raw.rename(
        columns={
            "Context": "Previous word",
            "Context.1": "Next word",
            "Keyword": "Segment",
        },
        inplace=True,
    )

    df_codes_raw = df_codes_raw.map(replace_apostrophes)
    df_context_raw = df_context_raw.map(replace_apostrophes)

    # merge raw csv files
    df_merged = df_codes_raw.merge(
        df_context_raw,
        how="left",
        on=["Document name", "Beginning", "Segment", "Document group"],
    )

    # if a word occurs more than once in the same line of the lyrics, duplicate rows will be created
    # using the unique Created column to drop these rows
    df_merged.drop_duplicates(subset="Created", keep="first", inplace=True)

    # if the first or final words of a file are coded, the pre or post contexts will be NaN, so we need to fill these with an empty string
    df_merged[df_merged.isna().any(axis=1)]
    df_merged.fillna("", inplace=True)

    # similarly, by default, the context columns include words from the previous/next line, but we want to get rid of these
    mask_pre_context = df_merged["Previous word"].str.endswith("  ")
    df_merged.loc[mask_pre_context, "Previous word"] = ""

    mask_post_context = df_merged["Next word"].str.startswith("  ")
    df_merged.loc[mask_post_context, "Next word"] = ""

    # rename segment column
    df_merged.rename(
        columns={
            "Segment": "Word",
        },
        inplace=True,
    )

    # unpack document name
    df_merged[["Artist", "Performance", "Song"]] = df_merged[
        "Document name"
    ].str.extract(r"(\w+)-(\w+)-([\w_]+)", expand=True)

    # unpack code
    df_merged[["Variable", "Value"]] = df_merged["Code"].str.split(" > ", expand=True)

    # rename performance context
    performance_contexts = {
        "so": "studio-original",
        "sc": "studio-cover",
        "lo": "live-original",
        "lc": "live-cover",
    }
    df_merged.replace({"Performance": performance_contexts}, inplace=True)

    # unpack performance context
    df_merged[["Type", "Performance"]] = df_merged["Performance"].str.split(
        "-", expand=True
    )

    # unpack time and social group
    df_merged[["Time", "Social group"]] = df_merged["Document group"].str.extract(
        r"(\d{4}s)_(\w+(?:_\w+)?)"
    )

    # make value column binary
    value_binary = {
        "a:": 1,
        "ai": 0,
        "t deletion": 1,
        "t realization": 0,
        "d deletion": 1,
        "d realization": 0,
        "r deletion": 1,
        "r realization": 0,
        "in": 1,
        "ing": 0,
        "ain't": 1,
        "isn't": 0,
        "s deletion": 1,
        "s realization": 0,
        "copula deletion": 1,
        "copula realization": 0,
    }
    df_merged.replace({"Value": value_binary}, inplace=True)

    # drop redundant columns
    df_merged.drop(
        columns=["Created", "Document name", "Code", "Beginning", "Document group"],
        inplace=True,
    )

    # prepare year df for merging
    df_year_raw = df_year_raw[["artist", "song", "year", "type"]]

    # merge to add year
    df_final = df_merged.merge(
        df_year_raw,
        how="left",
        left_on=["Artist", "Song", "Performance"],
        right_on=["artist", "song", "type"],
    )

    # drop redundant columns
    df_final.drop(columns=["artist", "song", "type"], inplace=True)

    # convert columns names to snake case
    df_final.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)

    # finalize column names
    df_final.rename(
        columns={
            "performance": "song_type",
            "type": "performance_type",
            "variable": "aae_feature",
            "value": "aae_realization",
        },
        inplace=True,
    )

    # export interim dataframe
    df_final.to_csv(
        "../../data/interim/corpus_studio_pre_processed.csv",
        index=False,
        encoding="UTF-8",
    )


if __name__ == "__main__":
    df_codes_raw = pd.read_csv("../../data/raw/corpus_studio.csv")
    df_context_raw = pd.read_csv("../../data/raw/corpus_studio_context.csv")
    df_year_raw = pd.read_csv("../../data/raw/corpus_year.csv")
    make_data(
        df_codes_raw=df_codes_raw,
        df_context_raw=df_context_raw,
        df_year_raw=df_year_raw,
    )
