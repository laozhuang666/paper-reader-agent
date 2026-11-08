import os
import json
import requests
from dotenv import load_dotenv
from retriever import PaperRetriever

# 加载环境变量
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"

# 初始化论文检索器
retriever = PaperRetriever(os.getenv("PDF_PATH", "paper.pdf"))

# 工具定义（Function Calling 格式）
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_paper",
            "description": "从论文中检索与问题相关的原文片段，返回带页码的内容。回答论文相关问题前必须调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要检索的关键词或问题"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def search_paper(query):
    """执行论文检索工具"""
    results = retriever.search(query)
    formatted = []
    for res in results:
        formatted.append(f"[第{res['page']}页] {res['content']}")
    return "\n\n".join(formatted)

def chat_completion(messages, use_tools=True):
    """调用 DeepSeek 接口，原生 requests 实现，附带错误详情打印"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1024
    }
    if use_tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    response = requests.post(BASE_URL, headers=headers, json=payload)
    if response.status_code != 200:
        print("\n=== 接口返回错误详情 ===")
        print(response.text)
        print("=======================\n")
    response.raise_for_status()
    return response.json()

def run_agent(user_query, messages_history):
    """Agent 主循环，修复工具调用消息格式"""
    messages = messages_history.copy()
    # 系统提示词
    system_prompt = "你是论文精读助手，回答必须基于论文原文，禁止编造内容。回答用户问题前，必须先调用 search_paper 工具检索原文。所有结论必须标注对应的页码，格式：(第X页)。如果检索不到相关内容，明确说明论文中未提及该内容。语言简洁专业，分点回答核心内容。"
    if not any(m["role"] == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": user_query})

    # 第一次调用，判断是否需要调用工具
    resp = chat_completion(messages)
    message = resp["choices"][0]["message"]

    # 修复：content 为 null 时替换为空字符串
    if message.get("content") is None:
        message["content"] = ""

    # 如果模型决定调用工具
    if "tool_calls" in message and message["tool_calls"]:
        # 关键修复：assistant 消息只添加 1 次，放在循环外面
        messages.append(message)

        # 循环执行每个工具调用，依次添加 tool 消息
        for tool_call in message["tool_calls"]:
            func_name = tool_call["function"]["name"]
            func_args = json.loads(tool_call["function"]["arguments"])

            # 执行工具
            if func_name == "search_paper":
                tool_result = search_paper(func_args["query"])

            # 添加对应 tool_call_id 的工具结果
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_result
            })

        # 第二次调用，模型根据工具结果生成最终回答
        final_resp = chat_completion(messages, use_tools=False)
        final_answer = final_resp["choices"][0]["message"]["content"]
        return final_answer, messages

    # 不需要调用工具，直接返回
    return message["content"], messages

if __name__ == "__main__":
    print("=" * 50)
    print("  论文精读 Agent 已启动，输入 exit 退出")
    print("=" * 50)
    history = []
    while True:
        query = input("\n请输入你的问题：")
        if query.lower() in ["exit", "quit"]:
            break
        try:
            answer, history = run_agent(query, history)
            print("\n回答：")
            print(answer)
        except Exception as e:
            print(f"\n运行出错：{e}")
