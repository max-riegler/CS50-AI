import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files_dict = dict()

    # Iterate through the files in the directory
    for file in os.listdir(directory):
        # Open each file, add the filename and the result of read() as a key-value pair to files_dict
        with open(os.path.join(directory, file), encoding="utf-8") as open_file:
            files_dict[file] = open_file.read()  

    return files_dict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # Tokenize document string using nltk
    tokenized = nltk.tokenize.word_tokenize(document.lower())
    # Construct a filtered list without punctuation (string.punctuation) and stopwords (nltk.corpus.stopwords.words("english"))
    filtered_list = [n for n in tokenized if n not in string.punctuation and n not in nltk.corpus.stopwords.words("english")]

    return filtered_list


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Empty dictionary for the idfs
    idfs = dict()
    # Total number of documents
    total_docs = len(documents)
    # Dictionary to count the number of documents containing each word
    doc_with_word = dict()
    # Iterate through documents looking at unique words in each
    for document in documents:
        unique_words = set(documents[document])
        for word in unique_words:
            if word not in doc_with_word:
                doc_with_word[word] = 1
            else:
                doc_with_word[word] += 1
    # Calculate idfs 
    for word in doc_with_word:
        idfs[word] = math.log((total_docs / doc_with_word[word]))

    return idfs

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Dictionary to hold scores for files
    scores = {file : 0 for file in files}
    # Iterate through all the words in query
    for word in query:
        # Iterate through words in the idfs dictionary
        if word in idfs:
            # Iterate through files and update scores with tf-idf
            for file in files:
              scores[file] += files[file].count(word) * idfs[word]
    # Sort the list ranked according to the scores
    sorted_list = sorted([file for file in files], key = lambda x : scores[x], reverse=True)

    return sorted_list[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # Dictionary to hold scores for sentences
    scores = {sentence : {"length" : 0, "query" : 0, "idf" : 0, "qtd" : 0} for sentence in sentences}

    # Iterate through sentences
    for sentence in sentences:
        scores[sentence]["length"] = len(nltk.word_tokenize(sentence))
        # Iterate through query set
        for word in query:
            # Update the score of a query word if it is in the sentence list
            if word in sentences[sentence]:
                scores[sentence]["idf"] += idfs[word]
                scores[sentence]["query"] += sentences[sentence].count(word)
        # Calculate the query term density
        scores[sentence]["qtd"] = scores[sentence]["query"] / scores[sentence]["length"]

    # Rank sentences by score and return n sentences
    sorted_list = sorted([sentence for sentence in sentences], key= lambda x: (scores[x]["idf"], scores[x]["qtd"]), reverse=True)

    # Return n entries for sorted list of sentences
    return sorted_list[:n]


if __name__ == "__main__":
    main()
