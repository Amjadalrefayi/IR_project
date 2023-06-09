# Search Engine Project
It's IR System for search about user's needs in a huge documents corpus.

## Features

- Train new models
- Search in your documents by your query
- Spelling correction for your query
- Predicted query list
- Evaluate information retrieval model system
- HTTP API
- UI


## Main Components

- Utils file have main function for load - save
- Preprocessor have text processing functionality 
- Http folder have API's

## IR Models
For now We support two types of indexing:
- TfIdf Index
- Word2Vec Model (Word Embedding)

## Setup

### Clone The Project
```sh
git clone https://github.com/Amjadalrefayi/IR_project.git
```

### Setup The Server
Install the libraries that exist in `requirements.txt`

```sh
python main.py
```

### API Endpoints
| Endpoint | Description                                                                 | Request | Response                                                                                 |
| ------ |-----------------------------------------------------------------------------| ------ |------------------------------------------------------------------------------------------|
| /search | Search for a documents by user's query                                      |  `{"query": "Your Query"}` | `{"data": {"document_id": "document content"}}` |
| /correct | Provide suggestion to correct user's query, it it contains a spelling error | `{"query": "Your Query"}` | `{"data": "Corrected Query"}`                                                            |
| /prediction | Retrieve prediction list                                                    | `{"query": "Your Query"}` | `{"data": [prediction1, prediction2]}`                                                      
| /documents/<document_id> | Retrive document content                                                    | URL param <document_id> | `{"data": "Document ID 1 Content"}`                       |

### Setup UI
To work with our UI you've to install Flutter and clone this project
```sh
git clone https://github.com/NajatiShehabi/IR_Project2023.git
```

[//]: # (## Train Your Model)

[//]: # (We already provide you with pre-trained models, you can find them in `documents-search-engine/ir_models` directory.)

[//]: # ()
[//]: # (But for new training you can run:)

[//]: # (- `documents-search-engine/engine/tfidf_notebook.ipynb` notebook for TfIdf Model.)

[//]: # (- `documents-search-engine/engine/word2_vec_notebook.ipynb` notebook for Word2Vec Model &#40;Word Embedding Model&#41;)

[//]: # ()
[//]: # (After running the notebook you will find the models in `$MODELS_DIRECTORY`.)

## Tech
- [Flask] - HTTP server library
- [Flutter] - For UI

## License

MIT

[//]: # (These are reference links used in the body)

   [Flask]: <https://flask.palletsprojects.com/en/2.3.x/installation/>
   [Flutter]: <https://docs.flutter.dev/get-started/install>
