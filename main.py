from flask import Flask
from forms import ASQSearchForm
from flask import flash, render_template, request, redirect
from absa2016_test import SentimentClassifier
import acd_laptops_test
import acd_restaurants_test
import re

app = Flask(__name__)
acd_laptops = acd_laptops_test.AspectCategoryClassifier()
acd_restaurants = acd_restaurants_test.AspectCategoryClassifier()
classifier = SentimentClassifier()

@app.route('/', methods=['GET', 'POST'])
def index():
    search = ASQSearchForm(request.form)
    if request.method == 'POST':
        return search_results(search)
 
    return render_template('index.html', form=search)

@app.route('/results')
def search_results(search):
    results = []
    search_string = search.data['search']
    domain = search.data['select']
    print(domain)

    if domain == 'lap':
        categories, polarities = classifier.test(search_string, acd_laptops, domain)
    else:
        categories, polarities = classifier.test(search_string, acd_restaurants, domain)

    print(categories)
    print(polarities)

    '''
    insert similartiy code here
    '''

    if not results:
        flash('No results found!')
        return redirect('/')

    else:
        # display results
        return render_template('results.html', results=results)
 
if __name__ == '__main__':
    app.run()