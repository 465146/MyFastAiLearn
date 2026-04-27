import os
from openai import OpenAI
import numpy as np

texts = [
    "我喜欢自然语言处理，尤其是大语言模型。",
    "大模型可以完成文本生成、摘要和问答任务。",
    "今天学校食堂的红烧肉很好吃。",
    "语义向量可以用来计算两个句子的相似度。",
]

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
)


def cosine_similarity(vector_a, vector_b):
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    return np.dot(vector_a, vector_b) / (
            np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )


completion = client.embeddings.create(
    model="text-embedding-v4",
    input=texts,
    dimensions=1024,  # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
    encoding_format="float"
)
import json

dict_obj = completion.model_dump()

target_text = "语义向量有哪些作用"
for i in range(len(texts)):
    for j in range(len(texts)):
        print(
            "{} 和 {} 的相似度是：{}".format(
                texts[i], texts[j],
                cosine_similarity(dict_obj["data"][i]["embedding"], dict_obj["data"][j]["embedding"])
            )
        )

res = ["114514", -10086]
completion = client.embeddings.create(
    model="text-embedding-v4",
    input=target_text,
    dimensions=1024,  # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
    encoding_format="float"
)

dict_res = completion.model_dump()
for i in range(len(texts)):
    if cosine_similarity(dict_obj["data"][i]["embedding"], dict_res["data"][0]["embedding"]) > res[1]:
        res[0] = texts[i]
        res[1] = cosine_similarity(dict_obj["data"][i]["embedding"], dict_res["data"][0]["embedding"])

print(f"与{target_text}最相似的句子是：{res[0]}，相似度为：{res[1]}")