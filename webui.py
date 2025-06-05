import os
import streamlit as st
from langchain_openai import ChatOpenAI
import json
from pathlib import Path

from determine_database import determine_database
from agent import agent


MERGED_JSON_PATH = 'papers.json'  # 替换为实际路径

# ============ 页面设置 ============
st.set_page_config(page_title="Agent", layout="wide")
st.title("🔍 Kerr Frequency Combs Q&A System")

# ====== 加载数据 ======
@st.cache_data
def load_paper_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_file_names(data_dict):
    return list(data_dict.keys())

paper_data = load_paper_data(MERGED_JSON_PATH)
file_names = load_file_names(paper_data)


# ============ 初始化 session_state ============
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "base_url" not in st.session_state:
    st.session_state.base_url = ""
if "weaviate_url" not in st.session_state:
    st.session_state.weaviate_url = st.secrets.get("WEAVIATE_URL", "")
if "weaviate_key" not in st.session_state:
    st.session_state.weaviate_key = st.secrets.get("WEAVIATE_KEY", "")
if "huggingface_key" not in st.session_state:
    st.session_state.huggingface_key = st.secrets.get("HUGGINGFACE_KEY", "")
if "model" not in st.session_state:
    st.session_state.model = "qwen-max-latest"
if "history" not in st.session_state:
    st.session_state.history = []  # 存储历史对话
if "parsed_output" not in st.session_state:
    st.session_state.parsed_output = None
if "result" not in st.session_state:
    st.session_state.result = {}
if "ori_answer" not in st.session_state:
    st.session_state.ori_answer = None
if "answer" not in st.session_state:
    st.session_state.answer = None
if "answer_graph" not in st.session_state:
    st.session_state.answer_graph = []


# ============ 设置历史记录 JSON 文件路径 ============
HISTORY_FILE = Path("chat_history.json")

# 从文件加载已有历史（首次运行加载）
if not st.session_state.history and HISTORY_FILE.exists():
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        try:
            st.session_state.history = json.load(f)
        except json.JSONDecodeError:
            st.session_state.history = []


# ============ 🔧 设置区域（侧边栏） ============
with st.sidebar:
    with st.expander("🔧 Setting", expanded=True):
        st.session_state.api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.api_key)
        st.session_state.base_url = st.text_input("Base URL", value=st.session_state.base_url)
        st.session_state.weaviate_url = st.text_input("Weaviate URL", value=st.session_state.weaviate_url)
        st.session_state.weaviate_key = st.text_input("Weaviate API Key", type="password", value=st.session_state.weaviate_key)
        st.session_state.huggingface_key = st.text_input("Huggingface API Key", value=st.session_state.huggingface_key)

        model_options = ['qwen-max-2025-01-25', 'qwen-max', 'qwen-max-latest', 'qwen-max-0428', "deepseek-v3"]  # 添加可选模型
        st.session_state.model = st.selectbox("Model Selection", model_options, index=model_options.index(st.session_state.model))

        show_answer = st.checkbox("Display Final Answer")
        show_context = st.checkbox("Display Reference", value=True)

    with st.expander("🔍 Search refs", expanded=False):     
        selected_paper_prefix = st.text_input("Title", key="prefix_input")
        matched_names = []
        if selected_paper_prefix.strip():
            matched_names = [name for name in file_names if name.lower().startswith(selected_paper_prefix.lower())][:20]
        if matched_names:
            selected_paper = st.selectbox("Select Titles", matched_names, key="paper_select")
        else:
            selected_paper = None
            st.info("Please enter at least a few characters to match the name of the literature (up to the first 20 results )")
        
        ref_index = st.text_input("Num of refs（such as '1'）", key="ref_input")

        if st.button("Search", key="confirm_btn") and selected_paper:
            paper = paper_data[selected_paper]

            # --- 查找引用 ---
            if ref_index.strip():
                references = paper.get("references", [])
                ref_dict = {}
                for ref in references:
                    ref_dict.update(ref)
                ref_content = ref_dict.get(ref_index)
                if ref_content:
                    st.markdown(f"**📚 refs: [{ref_index}]**")
                    st.markdown(ref_content)
                else:
                    st.warning(f"The reference numbered [{ref_index}] was not found")


# ============ 💬 主内容区 ============
st.subheader("💬 Please Enter Your Question:")
question = st.text_area("Question：", height=150, placeholder="Such as：What are the limitations of Kerr frequency combs ...")

if st.button("Question submit") and question.strip():
    if not st.session_state.api_key or not st.session_state.base_url:
        st.error("Please input API Key and Base URL in the sidebar.")
    else:
        with st.spinner(f"dealing with `{st.session_state.model}`..."):
            os.environ["OPENAI_API_KEY"] = st.session_state.api_key
            try:
                # 初始化模型
                llm = ChatOpenAI(model=st.session_state.model, base_url=st.session_state.base_url)

                weaviate_url = st.session_state.weaviate_url
                weaviate_api = st.session_state.weaviate_key
                huggingface_key = st.session_state.huggingface_key

                # 调用你的函数
                parsed_output, response = determine_database(question, llm)
                result = agent(llm, question, parsed_output, weaviate_url, weaviate_api, huggingface_key)

                st.markdown("""
                            <script type="text/javascript"
                                src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
                            </script>
                        """, unsafe_allow_html=True)

                question_type = parsed_output['question_type']

                if question_type == 'Knowledge-Type':
                    st.session_state.parsed_output = parsed_output
                    st.session_state.ori_answer = result['text']
                    st.session_state.answer = result['answer']
                    st.session_state.result = result['result']
                    
                    new_record = {
                        "question": question,
                        "ori answer": result['text'],
                        "answer": result['answer'],
                    }
                    
                    # 避免重复添加（例如刷新页面后再次添加）
                    if not any(record["question"] == question for record in st.session_state.history):
                        st.session_state.history.append(new_record)

                        # 保存到 JSON 文件
                        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                            json.dump(st.session_state.history, f, ensure_ascii=False, indent=4)

                    st.subheader("📊 Agent Result")

                    for idx, item in enumerate(st.session_state.result):
                        if idx == 0:
                            st.markdown(item['ori answer'])
                        else:
                            st.markdown(item['answer'], unsafe_allow_html=True)
                            if show_context and "details" in item:
                                with st.expander("🔍 References of the answer"):
                                    for _, detail in enumerate(item['details']):
                                        st.markdown("Reference:")
                                        st.markdown(detail.get('context'), unsafe_allow_html=True)
                                        st.markdown(f"Source title of the reference: {detail.get('title')}")
                                        if detail['analyze']:
                                            st.markdown(f"Analyze{detail.get('analyze', '')}")
                                            
        
                    if show_ori and st.session_state.ori_answer:
                        st.subheader("📊 Original Answer(Only for Text Database)")
                        st.text_area("", value=result['text'], height=400)

                    if show_answer and st.session_state.answer:
                        st.subheader("📊 Answer")
                        st.markdown(result['answer'])

                    if show_result and st.session_state.result:
                        with st.expander("Json Result"):
                            st.subheader("📊 Result List:")
                            st.json(result['result'])

               
                elif question_type == "Entity-Type":
                    st.session_state.answer_graph = result["answer"]
                    entities = result["extract"]
                    corrected = result["entities"]
                    cypher = result["cypher"]
                    retrieved = result['context']

                    st.subheader("📊 Agent Result")

                    st.markdown(result['answer'])
                    
                    if show_context:
                        with st.expander("📊 Retrieved Result"):
                            st.json(retrieved)

                elif question_type == "Mixed-Type":
                    response = result["response"]
                    st.subheader("📊 Agent Result")
                    st.markdown(response, unsafe_allow_html=True)

                    new_record = {
                        "question": question,
                        "answer": response,
                    }

                    if not any(record["question"] == question for record in st.session_state.history):
                        st.session_state.history.append(new_record)

                        # 保存到 JSON 文件
                        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                            json.dump(st.session_state.history, f, ensure_ascii=False, indent=4)

                    parsed = result["parsed"]
                    strategy = parsed["call_strategy"]
                    with st.expander("📚 Database Call Strategy"):
                        st.markdown(strategy)

                    result_list = result["result"]

                    if show_context:
                        for item in result_list:
                            db = item["database"]
                            step = item["step"]
                            st.subheader(f"Step {step}: Retrieved Results of {db}:")
                            if db == "Literature Text Database":
                                with st.expander("References of the answer:"):
                                    for part in item["contexts"]:
                                        st.markdown(part.get('chunk'), unsafe_allow_html=True)
                                        st.markdown(f"Source Title of the reference: {part.get('title')}")
                                    

                            elif db == "Literature Graph Database":
                                information = item['contexts']
                                if information:
                                    with st.expander("📊 Retrieved Result"):
                                        st.json(information['context'])
                                else:
                                    st.markdown("No Relative Papers Found")

                    st.session_state.result = result_list
    
                    if show_result and st.session_state.result:
                            with st.expander("Json Result"):
                                st.subheader("📊 Result List:")
                                st.json(result_list)
                                    
            except Exception as e:
                st.error(f"Error：{e}")


# ============ 🐞 调试信息展示 ============
if show_debug and st.session_state.parsed_output:
    st.subheader("🛠 Question Type Selection")
    st.json(st.session_state.parsed_output)

# ============ 清空历史记录按钮 ============
if st.button("🧹 Remove all records"):
    st.session_state.history = []
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()
    st.success("The historical records have been cleared!")

# ============ 📜 历史记录展示 ============
if st.session_state.history:
    st.subheader("🕘 历史对话记录")
    for i, record in enumerate(reversed(st.session_state.history[-5:]), 1):
        with st.expander(f"历史记录 {i}"):
            st.markdown(f"**Q:** {record['question']}")
            st.markdown(f"**A:** {record['answer']}")
