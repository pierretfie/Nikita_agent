from llama_cpp import Llama

llm = Llama(model_path="/home/eclipse/Nikita_Agent_model/mistral.gguf")

response = llm("Say: 'Hello from Nikita AI'")
print(response["choices"][0]["text"].strip())
