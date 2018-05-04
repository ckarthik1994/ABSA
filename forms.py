# forms.py
 
from wtforms import Form, StringField, SelectField, validators
 
class ASQSearchForm(Form):
    choices = [('lap', 'Laptops'),
               ('rest', 'Restaurants')]
    select = SelectField('Select Domain:', choices=choices)
    search = StringField('')