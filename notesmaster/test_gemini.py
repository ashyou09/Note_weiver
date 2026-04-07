import requests

url = "http://localhost:7860/api/generate"
data = {"mode":"topic", "topic":"testing gemini model again", "model":"gemini-2.0-flash"}

try:
    with requests.post(url, json=data, stream=True) as r:
        for line in r.iter_lines():
            if line:
                print(line.decode('utf-8'))
except Exception as e:
    print(f"Error: {e}")
