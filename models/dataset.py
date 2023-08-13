import hashlib
import json

def sha1_hash(text):
    sha1 = hashlib.sha1()
    sha1.update(text.encode('utf-8'))
    return sha1.hexdigest()

data = [
    {"original": "hello", "hashed": sha1_hash("hello")},
    {"original": "world", "hashed": sha1_hash("world")},
    {"original": "fullzer4", "hashed": sha1_hash("fullzer4")}
]

output_file = 'dataset.json'

with open(output_file, 'w') as json_file:
    json.dump(data, json_file, indent=4)