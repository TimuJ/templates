#Attempts at delpoying model with [docker](https://www.docker.com/) + [fastapi](https://fastapi.tiangolo.com/) 

## How to run
* Build Docker image using `docker build . -t text_classifier`
* Run Docker container using `docker run --rm -it -p 19:19 text_classifier`
* Go to `http://127.0.0.1:19/docs` to see all available methods of the API

##Source code
*[server.py](server.py) contains API logic
*[model.joblib](model.joblib) dummy model for text classif
*[tfidf.pkl](tfidf.pkl) contains text for tfidf transformation
*[Dockerfile](Dockerfile) describes a Docker image that is used to run the API


