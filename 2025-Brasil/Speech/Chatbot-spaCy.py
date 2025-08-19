import spacy

nlp = spacy.load("en_core_web_sm")

user_input = input("Enter your text: ")
doc = nlp(user_input)

entities = doc.ents

for entity in entities:
    print(entity.text, entity.label_, entity.start_char)