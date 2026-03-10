"""
Streamlit MVP 前端

快速验证端到端效果：
- 左侧边栏：系统状态、配置面板
- 主区域：对话界面（流式输出 + 引用来源展示）
- 底部：故障码快捷查询
"""

import json
import re
import time

import requests
import streamlit as st

# ─── 配置 ────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000/api/v1"
PAGE_TITLE = "新能源汽车故障诊断助手"


# ─── 页面设置 ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔧 新能源汽车故障诊断智能问答系统")
st.caption("基于 Advanced RAG + 知识图谱 | 数字老师傅")


# ─── 侧边栏 ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("系统状态")

    # 健康检查
    try:
        resp = requests.get(f"{API_BASE.replace('/api/v1', '')}/health", timeout=3)
        if resp.status_code == 200:
            st.success("API 服务正常")
        else:
            st.error(f"API 异常: {resp.status_code}")
    except Exception:
        st.error("API 服务未启动")
        st.info("请运行：\n```bash\nuvicorn src.api.main:app --port 8000\n```")

    st.divider()
    st.header("快捷故障码查询")

    quick_code = st.text_input("输入故障码", placeholder="如 P0300", key="quick_code")
    if st.button("查询", key="quick_btn") and quick_code:
        st.session_state["pending_query"] = quick_code.upper().strip()

    st.divider()
    st.header("关于")
    st.markdown("""
    **技术栈**
    - LLM: Qwen2-VL-7B-Instruct
    - 检索: BM25 + BGE Hybrid
    - 图谱: Neo4j 5
    - 评估: RAGAS
    """)


# ─── 会话状态初始化 ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# ─── 显示历史消息 ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("查看引用来源"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**[{src['ref_id']}]** {src['source']} 第{src['page']}页"
                        + (f" · {src['chapter']}" if src.get('chapter') else "")
                    )
                    st.caption(src.get("snippet", ""))


# ─── 发送查询 ──────────────────────────────────────────────────────────
def send_query(query: str):
    """发送查询到 API 并流式显示结果。"""
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        sources = []

        try:
            with requests.post(
                f"{API_BASE}/chat/stream",
                json={
                    "message": query,
                    "session_id": st.session_state.session_id,
                },
                stream=True,
                timeout=120,
            ) as resp:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8")
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        if "token" in data:
                            full_text += data["token"]
                            placeholder.markdown(full_text + "▌")
                        elif data.get("done"):
                            sources = data.get("sources", [])
                            if not st.session_state.session_id:
                                st.session_state.session_id = data.get("session_id")
                        elif "error" in data:
                            st.error(f"错误: {data['error']}")
                            return

        except requests.exceptions.ConnectionError:
            st.error("无法连接到 API 服务，请确认服务已启动")
            return
        except Exception as e:
            st.error(f"请求失败: {e}")
            return

        placeholder.markdown(full_text)

        if sources:
            with st.expander("查看引用来源"):
                for src in sources:
                    st.markdown(
                        f"**[{src['ref_id']}]** {src['source']} 第{src['page']}页"
                        + (f" · {src['chapter']}" if src.get('chapter') else "")
                    )
                    st.caption(src.get("snippet", ""))

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_text,
        "sources": sources,
    })


# ─── 处理 pending query（来自侧边栏快捷查询）──────────────────────────
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None
    send_query(query)

# ─── 主输入框 ─────────────────────────────────────────────────────────
if user_input := st.chat_input("输入故障码（如 P0300）或故障现象描述..."):
    send_query(user_input)
