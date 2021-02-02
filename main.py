import re
from pprint import pprint

import requests
import spacy
from IPython.core.display import display, HTML
from bs4 import BeautifulSoup
from spacy import displacy
from collections import Counter
import en_core_web_sm

nlp = en_core_web_sm.load()

"""
https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da
"""

ex = "European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in" \
     " the mobile phone market and ordered the company to alter its practices"

# Apply labels on the entity level
doc = nlp(ex)


# print("Entity-level annotation")
# pprint([(X.text, X.label_) for X in doc.ents])

# Token-level entity annotation using BILUO tagging scheme:
# BEGIN - the first token of a multi-token entity
# IN - an inner token of a multi-token entity
# LAST - the final token of a multi-token entity
# UNIT - a single-token entity
# OUT - a non-entity token

# print("\nToken-level annotation")
# pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])

# "B" means the token begins an entity, "I" means it is inside an entity, "O" means it is outside an entity,
# and "" means no entity tag is set.


def url_to_string(url):
    res = requests.get(url)
    html = res.text
    # soup = BeautifulSoup(html, 'html5lib')
    soup = BeautifulSoup(html, 'html.parser')
    for script in soup(["script", "style", 'aside']):
        script.extract()
    return " ".join(re.split(r'[\n\t]+', soup.get_text()))


url = "https://www.nytimes.com/2018/08/13/us/politics/peter-strzok-fired-fbi.html?hp&action=click&pgtype=Homepage" \
      "&clickSource=story-heading&module=first-column-region&region=top-news&WT.nav=top-news"
ny_bb = url_to_string(url)
article = nlp(ny_bb)
print(len(article.ents))

# for ent in article.ents:
#     print(type(ent), ent)

# Labels of found entities
print("Labels of found entities:")
labels = [x.label_ for x in article.ents]
print(Counter(labels))

# The first 3 most frequent tokens
print("The first 3 most frequent tokens")
items = [x.text for x in article.ents]
print(Counter(items).most_common(3))

sentences = [x for x in article.sents]
print(sentences[28])

displacy.render(nlp(str(sentences[20])), jupyter=True, style='ent')

displacy.render(nlp(str(sentences[20])), style='dep', jupyter=True, options={'distance': 120})

# doc1 = nlp("This is a sentence.")
# doc2 = nlp("This is another sentence.")
# html = displacy.render([doc1, doc2], style="dep", page=True)
# display(HTML(html))

# display(HTML(displacy.render(nlp(str(sentences[20])), style='dep', options={'distance': 120})))

# verbatim, extract part-of-speech and lemmatize this sentence.
l = [(x.orth_, x.pos_, x.lemma_) for x in [y
                                           for y
                                           in nlp(str(sentences[28]))
                                           if not y.is_stop and y.pos_ != 'PUNCT']]

