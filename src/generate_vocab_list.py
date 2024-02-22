import os
from collections import Counter
from docx import Document
import gzip
import re


def generate_vocab_list():
    """
    function to create a vocab list which can be used by wordninja to parse words
    in "Next word" and "Previous word" columns of dataset that are not separated by
    a space due to a bug in MAXQDA
    """
    vocabulary = []

    # Loop through each file in the folder
    for filename in os.listdir("../data/raw/transcriptions"):
        if filename.endswith(".docx"):
            file_path = os.path.join("../data/raw/transcriptions", filename)

            # Read the document
            doc = Document(file_path)

            # Extract text from the document and process it
            for paragraph in doc.paragraphs:
                text = paragraph.text

                # Split text into words
                words = text.split()

                # Process each word
                for word in words:
                    # Handle contractions
                    if "'" in word:
                        match = re.match(r"(\w+)'(\w+)", word)
                        if match:
                            base_word = match.group(1)
                            contraction = "'" + match.group(2)
                            vocabulary.append(base_word)
                            vocabulary.append(contraction)
                    else:
                        word = word.replace("(", "").replace(")", "")
                        vocabulary.append(word)

    # Count occurrences of each word
    word_counts = Counter(vocabulary)

    # Sort vocabulary by word counts
    sorted_vocabulary = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    with open("../data/interim/vocab_list.txt", "w") as fp:
        for item in sorted_vocabulary:
            # write each item on a new line
            fp.write(f"{item[0]}\n")
        print("Done")

    # Create gzip file
    with open("../data/interim/vocab_list.txt", "rb") as f_in, gzip.open(
        "../data/interim/vocab_list.txt.gz", "wb"
    ) as f_out:
        f_out.writelines(f_in)
