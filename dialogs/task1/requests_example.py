import requests

url = "http://localhost:8000/SentimentClassifier/"

params = {
    "text": "I really enjoyed the movie. It was fantastic!"
}

response = requests.post(url, json=params)
data = response.json()


params = [
        {"text": "This product is great!"},
        {"text": "I'm not satisfied with the service."}
]

response = requests.post(url, json=params)
data = response.json()
print(data)


params = {
    "text": [
        "This is the first text.",
        "And this is the second text."
    ]
}


response = requests.post(url, json=params)
data = response.json()
print(data)