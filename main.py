from flask import Flask
from forms import ASQSearchForm
from flask import flash, render_template, request, redirect
from flask_table import Table, Col
from absa2016_test import SentimentClassifier
import acd_laptops_test
import acd_restaurants_test
import re
import MongoDB_Helper as MongoDB_Helper

app = Flask(__name__, static_url_path='/static')
acd_laptops = acd_laptops_test.AspectCategoryClassifier()
acd_restaurants = acd_restaurants_test.AspectCategoryClassifier()
classifier = SentimentClassifier()


class ItemTable(Table):
    reviewText = Col('Review Text')
    sentimentLabels = Col('Aspect Category Sentiments')

@app.route('/<path:path>')
def static_file(path):
    return app.send_static_file(path)

@app.route('/', methods=['GET', 'POST'])
def index():
    search = ASQSearchForm(request.form)
    if request.method == 'POST':
        return search_results(search)
 
    return render_template('search.html', form=search)

@app.route('/results')
def search_results(search):
    results = []
    search_string = search.data['search']
    domain = search.data['selectDomain']
    print(search.data)
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
        item = "("+cat.upper()+","+pol.lower()+")"
        opinions.append(item)

    method = search.data['selectMethod']
    if method == "naive":
        search_results = MongoDB_Helper.GetSimilarDocuments(opinions)
        ranked_results = MongoDB_Helper.CosineSimilarity(search_results,opinions)
    else:
        ranked_results = MongoDB_Helper.GetTopDocumentsTFIDF(opinions)

    if len(ranked_results)==0:
        flash('No results found!')
        return redirect('/')

    searchResultItems = []
    for i in range(0, len(ranked_results)):
        opinions = ranked_results[i]['Opinions']
        reviewContent = ranked_results[i]['Review']
        searchResultItems.append(dict(reviewText=reviewContent, sentimentLabels=opinions))

    table = ItemTable(searchResultItems, border=1, classes=["resultsTable"])
    tableHTML = table.__html__()

    responseHTML = ""
    with open ("templates/results.html", "r") as myfile:
        responseHTML=myfile.readlines()

    responseHTML = ''.join(responseHTML)
    responseHTML = responseHTML.replace("$table$", tableHTML)

    return responseHTML
 
if __name__ == '__main__':
    app.run()