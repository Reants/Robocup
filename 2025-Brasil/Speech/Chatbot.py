import ollama
respuesta = ollama.chat(model='mistral', messages=[
    {'role': 'user', 'content': '¿Cuál es la capital de Colombia?'}
])
print(respuesta['message']['content'])

