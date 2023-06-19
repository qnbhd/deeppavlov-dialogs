import requests

url = "http://localhost:8000/SentimentClassifier/"

params = {
    "texts": ["I really enjoyed the movie. It was fantastic!"]
}

response = requests.post(url, json=params)
data = response.json()
print(data)