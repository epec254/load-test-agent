import openai

client = openai.OpenAI(
    base_url="http://localhost:4000",
    api_key="sk-1234567890",
)

stream = client.chat.completions.create(
    model="fast-cheap",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
