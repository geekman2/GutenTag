from spacy.en import English


def tokenize(doc, parser=English()):
    # Intialize the parser and tokenize the text retrieved
    return parser(doc)
