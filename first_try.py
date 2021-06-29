import nltk
from nltk.corpus import treebank

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags, ne_chunk
from pprint import pprint

# Run this once:
# nltk.download()

ex = "European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone " \
"market and ordered the company to alter its practices."


def preprocess(sent):
    # word tokenization and part-of-speech tagging
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent


sent = preprocess(ex)
print("TOKENS: ", sent)

pattern = 'NP: {<DT>?<JJ>*<NN>}'  # Noun Phrase

cp = nltk.RegexpParser(pattern)
cs = cp.parse(sent)
print("Chunk parser of noun phrases: \n", cs)

# Inside-Outside-Beginning tags
print("Applying IOB tags...")
iob_tagged = tree2conlltags(cs)
pprint(iob_tagged)

# Named entities recognition using a classifier
print("Applying classifier to recognize the named entities..")  # GPE=Global Political Economy
ne_tree = ne_chunk(pos_tag(word_tokenize(ex)))
print(ne_tree)
# entities = nltk.chunk.ne_chunk(sent)
# pprint(entities)
#
# t = treebank.parsed_sents('wsj_0001.mrg')[0]
# t.draw()
