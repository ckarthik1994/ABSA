from flask import Flask
from forms import ASQSearchForm
from flask import flash, render_template, request, redirect
from flask_table import Table, Col
from absa2016_test import SentimentClassifier
import acd_laptops_test
import acd_restaurants_test
import re
import MongoDB_Helper as MongoDB_Helper

app = Flask(__name__)
acd_laptops = acd_laptops_test.AspectCategoryClassifier()
acd_restaurants = acd_restaurants_test.AspectCategoryClassifier()
classifier = SentimentClassifier()

class ItemTable(Table):
    reviewText = Col('Review Text')
    sentimentLabels = Col('Aspect Category Sentiments')

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

    opinions = []
    for i in range(0, len(categories)):
        cat = categories[i]
        pol = polarities[i]
        item = "("+cat+","+pol+")"
        opinions.append(item)

    search_results = MongoDB_Helper.GetSimilarDocuments(opinions)
    ranked_results = MongoDB_Helper.CosineSimilarity(search_results,opinions)

    if len(ranked_results)==0:
        flash('No results found!')
        return redirect('/')

    searchResultItems = []
    for i in range(0, len(ranked_results)):
        opinions = ranked_results[i]['Opinions']
        reviewContent = ranked_results[i]['Review']
        searchResultItems.append(dict(reviewText=reviewContent, sentimentLabels=opinions))

    table = ItemTable(searchResultItems, border=1)
    tableHTML = table.__html__()

    responseHTML = '<html><head><title>ASQ - Aspect Sentiment Query</title></head><body><h2>ASQ us what you want ;)</h2>'+tableHTML+'</body></html>'

    return responseHTML
 
if __name__ == '__main__':
    app.run()