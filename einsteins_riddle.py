import re
from collections import Counter

import en_core_web_sm
import requests
from bs4 import BeautifulSoup
from spacy import displacy

nlp = en_core_web_sm.load()


def url_to_string(url):
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, 'html.parser')
    for script in soup(["script", "style", 'aside']):
        script.extract()
    return " ".join(re.split(r'[\n\t]+', soup.get_text()))


url = "https://udel.edu/~os/riddle.html#:~:text=Einstein's%20riddle&text=There%20are%205%20houses%20in" \
      ",or%20drink%20the%20same%20beverage."
einstein_riddle_text = url_to_string(url)
article = nlp(einstein_riddle_text)

displacy.render(article, jupyter=True, style='ent')

print("Labels of found entities:")
labels = [x.label_ for x in article.ents]
print(Counter(labels))

print("Found entities by their types:")
entities = [(x.text, x.label_) for x in article.ents]
print(Counter(entities))


