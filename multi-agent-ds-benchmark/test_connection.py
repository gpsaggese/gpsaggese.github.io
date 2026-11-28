from config import get_llm

print("Connecting to NVIDIA API...")
llm = get_llm()
response = llm.invoke("Say hello and confirm you are working. Keep it to one sentence.")
print("Response:", response.content)
print("Connection successful!")
