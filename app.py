import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
from starlette.responses import FileResponse
import uuid
from typing import Dict, List, Optional


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None



# 初始化 FastAPI 应用
app = FastAPI(title="Customer Service Bot")

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

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

# 存储会话历史的字典 {session_id: [messages]}
conversation_history: Dict[str, List[dict]] = {}


# 定义请求数据模型
class ChatRequest(BaseModel):
    question: str
    session_id: str = None  # 可选的会话ID


# 定义响应数据模型
class ChatResponse(BaseModel):
    answer: str
    session_id: str


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

        # 生成或使用现有的会话ID
        session_id = request.session_id or str(uuid.uuid4())

        # 如果这是新会话，初始化系统消息
        if session_id not in conversation_history:
            conversation_history[session_id] = [
                {
                    "role": "system",
                    "content": "你是自然语言处理课程助教，回答要准确、简洁。",
                }
            ]

        # 添加用户消息到历史记录
        conversation_history[session_id].append({
            "role": "user",
            "content": request.question,
        })

        # 调用大模型获取回答（包含完整的对话历史）
        completion = client.chat.completions.create(
            model=QWEN_CHAT_MODEL,
            messages=conversation_history[session_id],
            temperature=0.3,
        )

        answer = completion.choices[0].message.content

        # 将助手的回复也添加到历史记录中
        conversation_history[session_id].append({
            "role": "assistant",
            "content": answer,
        })

        return ChatResponse(answer=answer, session_id=session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


@app.delete("/chat/{session_id}")
async def clear_conversation(session_id: str):
    """清除指定会话的历史记录"""
    if session_id in conversation_history:
        del conversation_history[session_id]
        return {"message": f"会话 {session_id} 已清除"}
    else:
        raise HTTPException(status_code=404, detail="会话不存在")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
