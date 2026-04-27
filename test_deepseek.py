import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

completion = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {
            "role": "system",
            "content": "你是自然语言处理课程助教，回答要准确、简洁。",
        },
        {
            "role": "user",
            "content": "请用三句话解释什么是自然语言处理。",
        },
    ],
    stream=False,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}}
)
answer = completion.choices[0].message.content
print(answer)