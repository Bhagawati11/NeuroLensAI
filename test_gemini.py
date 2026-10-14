
import os
from google import genai

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="Reply with exactly: Gemini API working"
)

print(response.text)
