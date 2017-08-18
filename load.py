import spacy
from spacy.symbols import nsubj, VERB


def disable_parser():
    nlp = spacy.load('en', parser=False)


def part_of_speech():
    nlp = spacy.load('en')                 
    doc = nlp(u'They told us to duck.')
    for word in doc:
        print(word.text, word.lemma, word.lemma_, word.tag, word.tag_, word.pos, word.pos_)

def dependency_parsing():
    # Finding a verb with a subject from below — good
    nlp = spacy.load('en')                 
    doc = nlp(u'They told us to duck.Because the syntactic relations form a tree, every word has exactly one head. ')
    verbs = set()
    for possible_subject in doc:
        if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
            verbs.add(possible_subject.head)
    print (verbs)

def named_entity():
    nlp = spacy.load('en')  
    doc = nlp(u'London is a big city in the United Kingdom.')
    print(doc[0].text, doc[0].ent_iob, doc[0].ent_type_)
    # (u'London', 2, u'GPE')
    print(doc[1].text, doc[1].ent_iob, doc[1].ent_type_)
    # (u'is', 3, u'')

    #annotation:
    #The easiest way to set entities is to assign to the doc.ents attribute.
    doc = nlp(u'London is a big city in the United Kingdom.')
    doc.ents = []
    assert doc[0].ent_type_ == ''
    doc.ents = [Span(doc, 0, 1, label=doc.vocab.strings['GPE'])]
    assert doc[0].ent_type_ == 'GPE'
    doc.ents = []
    doc.ents = [(u'LondonCity', doc.vocab.strings['GPE'], 0, 1)]

def word_vector():
    nlp = spacy.load('en')  
    apples, and_, oranges = nlp(u'apples and oranges')
    print(apples.vector.shape)
    # (1,)
    apples.similarity(oranges)








