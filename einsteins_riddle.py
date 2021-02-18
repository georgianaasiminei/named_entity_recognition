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
# article = nlp(einstein_riddle_text)

riddle_text = """
There are 5 houses in five different colors.
In each house lives a person with a different nationality.
These five owners drink a certain type of beverage, smoke a certain brand of cigar and keep a certain pet.
No owners have the same pet, smoke the same brand of cigar or drink the same beverage.
The question is: Who owns the fish?

Hints
the Brit lives in the red house
the Swede keeps dogs as pets
the Dane drinks tea
the green house is on the left of the white house
the green house's owner drinks coffee
the person who smokes Pall Mall rears birds
the owner of the yellow house smokes Dunhill
the man living in the center house drinks milk
the Norwegian lives in the first house
the man who smokes blends lives next to the one who keeps cats
the man who keeps horses lives next to the man who smokes Dunhill
the owner who smokes BlueMaster drinks beer
the German smokes Prince
the Norwegian lives next to the blue house
the man who smokes blend has a neighbor who drinks water"""

article = nlp(riddle_text)

displacy.render(article, jupyter=True, style='ent')

print("Labels of found entities:")
labels = [x.label_ for x in article.ents]
print(Counter(labels))

print("Found entities by their types:")
entities = [(x.text, x.label_) for x in article.ents]
print(Counter(entities))


