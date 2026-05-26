from flask import Flask, render_template, request, jsonify

from rag import search_cars
from model import check_price

app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return render_template('index.html')


# SEARCH ENDPOINT
@app.route('/search', methods=['POST'])
def search():

    try:
        query = request.form.get('user_query')

        if not query:
            return "Query cannot be empty"

        # Get RAG search results
        results = search_cars(query)

        # Add AI price analysis to every car
        for car in results:

            price_result = check_price(
                car['make'],
                car['model'],
                int(car['year']),
                car['fuel'],
                car['transmission'],
                int(car['kms_driven']),
                float(car['price_lakhs'])
            )

            # Add new fields for frontend
            car['predicted_price'] = price_result['estimated_price']
            car['valuation'] = price_result['verdict']

        return render_template(
            'index.html',
            cars=results,
            query=query
        )

    except Exception as e:
        return f"Error: {str(e)}"


# PRICE CHECK ENDPOINT
@app.route('/price-check', methods=['POST'])
def price_check():

    try:
        data = request.json

        result = check_price(
            data['make'],
            data['model'],
            int(data['year']),
            data['fuel'],
            data['transmission'],
            int(data['kms_driven']),
            float(data['listed_price'])
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e)
        })


if __name__ == '__main__':
    app.run(debug=True)