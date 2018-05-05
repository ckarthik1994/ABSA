# forms.py
 
from wtforms import Form, StringField, SelectField, validators
 
class ASQSearchForm(Form):
    domainChoices = [('lap', 'Laptops'),
               		 ('rest', 'Restaurants')]
    selectDomain = SelectField('Select Domain:', choices=domainChoices)
    methodChoices = [('naive', 'Naive'),
    				 ('tfidf', 'Cosine Distance')]
    selectMethod = SelectField('Select Method:', choices=methodChoices)
    search = StringField('')