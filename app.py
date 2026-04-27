import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
from starlette.responses import FileResponse

# 初始化 FastAPI 应用
app = FastAPI(title="Customer Service Bot")

# 挂载静态文件目录
# app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/static",StaticFiles(directory="static"),name="static")

# 配置大模型客户端
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_CHAT_MODEL = "qwen-plus"

api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise RuntimeError("没有读取到环境变量 DASHSCOPE_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url=QWEN_BASE_URL,
)


# 定义请求数据模型
class ChatRequest(BaseModel):
    question: str


@app.get("/")
async def read_root():
    """返回主页"""
    return FileResponse("static/index.html")


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """处理聊天请求"""
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="问题不能为空")

        # 调用大模型获取回答
        completion = client.chat.completions.create(
            model=QWEN_CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是自然语言处理课程助教，回答要准确、简洁。",
                },
                {
                    "role": "user",
                    "content": request.question,
                },
            ],
            temperature=0.3,
        )

        answer = completion.choices[0].message.content
        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
