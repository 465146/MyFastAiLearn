import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
from starlette.responses import FileResponse
import uuid
from typing import Dict, List, Optional

# 初始化 FastAPI 应用
app = FastAPI(title="Customer Service Bot")

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 配置第一个 AI - NLP 课程助教 (DeepSeek)
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_CHAT_MODEL = "deepseek-v4-pro"

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    raise RuntimeError("没有读取到环境变量 DEEPSEEK_API_KEY")

deepseek_client = OpenAI(
    api_key=deepseek_api_key,
    base_url=DEEPSEEK_BASE_URL
)

# 配置第二个 AI - FastGPT 知识库助手
FASTGPT_BASE_URL = "https://cloud.fastgpt.cn/api/v1"
FASTGPT_API_KEY = os.getenv("FASTGPT_API_KEY")
FASTGPT_APP_ID = os.getenv("FASTGPT_APP_ID")

fastgpt_client = None
if FASTGPT_API_KEY and FASTGPT_APP_ID:
    fastgpt_client = OpenAI(
        api_key=FASTGPT_API_KEY,
        base_url=FASTGPT_BASE_URL
    )

# 存储会话历史的字典 {session_id: [messages]}
conversation_history: Dict[str, List[dict]] = {}


# 定义请求数据模型
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    ai_type: str = "nlp"  # "nlp" 或 "fastgpt"


# 定义响应数据模型
class ChatResponse(BaseModel):
    answer: str
    session_id: str
    ai_type: str


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

        # 验证 ai_type
        if request.ai_type not in ["nlp", "fastgpt"]:
            raise HTTPException(status_code=400, detail="无效的 AI 类型")

        # 如果使用 fastgpt 但没有配置，返回错误
        if request.ai_type == "fastgpt" and not fastgpt_client:
            raise HTTPException(status_code=500, detail="FastGPT 未配置，请检查环境变量")

        # 生成或使用现有的会话ID
        session_id = request.session_id or str(uuid.uuid4())

        # 为不同的 AI 类型创建独立的会话历史
        history_key = f"{session_id}_{request.ai_type}"

        # 如果这是新会话，初始化系统消息
        if history_key not in conversation_history:
            if request.ai_type == "nlp":
                conversation_history[history_key] = [
                    {
                        "role": "system",
                        "content": "你是自然语言处理课程助教，回答要准确、简洁。",
                    }
                ]
            else:  # fastgpt
                conversation_history[history_key] = [
                    {
                        "role": "system",
                        "content": "你是知识库助手，基于知识库内容回答问题。",
                    }
                ]

        # 添加用户消息到历史记录
        conversation_history[history_key].append({
            "role": "user",
            "content": request.question,
        })

        # 根据 AI 类型选择不同的客户端
        if request.ai_type == "nlp":
            completion = deepseek_client.chat.completions.create(
                model=DEEPSEEK_CHAT_MODEL,
                messages=conversation_history[history_key],
                temperature=0.3,
            )
        else:  # fastgpt
            completion = fastgpt_client.chat.completions.create(
                model=FASTGPT_APP_ID,  # FastGPT 使用 app_id 作为 model
                messages=conversation_history[history_key],
                temperature=0.3,
            )

        answer = completion.choices[0].message.content

        # 将助手的回复也添加到历史记录中
        conversation_history[history_key].append({
            "role": "assistant",
            "content": answer,
        })

        return ChatResponse(answer=answer, session_id=session_id, ai_type=request.ai_type)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


@app.delete("/chat/{session_id}")
async def clear_conversation(session_id: str):
    """清除指定会话的历史记录"""
    # 清除该 session_id 下的所有 AI 类型的历史记录
    keys_to_delete = [key for key in conversation_history.keys() if key.startswith(f"{session_id}_")]
    for key in keys_to_delete:
        del conversation_history[key]

    if keys_to_delete:
        return {"message": f"会话 {session_id} 的所有历史记录已清除"}
    else:
        raise HTTPException(status_code=404, detail="会话不存在")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
