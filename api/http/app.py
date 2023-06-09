from flask import Flask, jsonify, request
from flask_cors import CORS
from engine.tfidf import TfIdfEngine
# from engine.prediction import Prediction
app = Flask(__name__)
CORS(app)


@app.route("/search", methods=["POST"])
def search_by_query():
    body = request.json

    if 'query' not in body or len(body['query']) == 0:
        return jsonify({"error": "You have to pass query"})

    query = body['query']
    tfidf = TfIdfEngine()
    documents = tfidf.match_query(str(query))

    print(documents)

    result = {"data": {}}
    for doc in documents:
        result['data'][doc] = tfidf.get_file_content(doc)
    print(result)
    return jsonify(result)
    # return jsonify({'data': {"doc1": "when an unknown printer took a galley of type and scrambled it to make a type "
    #                                  "specimen book. It has survived not only five centuries", "doc2": "Reader will "
    #                                                                                                    "be distracted"
    #                                                                                                    " by the "
    #                                                                                                    "readable "
    #                                                                                                    "content of a "
    #                                                                                                    "page when "
    #                                                                                                    "looking at "
    #                                                                                                    "its layout."}})


@app.route("/prediction", methods=["POST"])
def search_by_prediction():
    body = request.json

    if 'query' not in body or len(body['query']) == 0:
        return jsonify({"error": "You have to pass query"})

    query = body['query']

    return jsonify({'data': ["There are many variations of passages of Lorem Ipsum available.", "Contrary to popular "
                                                                                                "belief, Lorem Ipsum "
                                                                                                "is not simply random "
                                                                                                "text. It has roots "
                                                                                                "45 BC."]})


@app.route("/documents/<document_id>")
def get_document_by_id(document_id: str):
    print(document_id)
    return jsonify({'data': "content"})


@app.route("/correct", methods=["POST"])
def get_correct():

    # prediction = Prediction()
    body = request.json

    if 'query' not in body or len(body['query']) == 0:
        return jsonify({"error": "You have to pass query"})

    query = body['query']

    return jsonify({'data': "Hello"})


def start_server():
    app.run(host="192.168.43.214")
