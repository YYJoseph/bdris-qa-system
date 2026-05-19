import gradio as gr
import pymupdf
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

# 初始化
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="qwen2.5:0.5b")
vectorstore = None

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += chunk_size - overlap
    return chunks

def upload_pdf(pdf_file):
    global vectorstore
    doc = pymupdf.open(pdf_file.name)
    chunks = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            for chunk in split_text(text):
                chunks.append(Document(
                    page_content=chunk,
                    metadata={"page": i+1}
                ))
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./bdris_db"
    )
    return f"✅ 成功加载 {len(doc)} 页，切分为 {len(chunks)} 个片段！"

# 定义工具
@tool
def search_paper(query: str) -> str:
    """从已上传的BD-RIS论文中检索相关内容"""
    if vectorstore is None:
        return "请先上传论文！"
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([d.page_content for d in docs])

@tool
def calculate(expression: str) -> str:
    """计算数学表达式，如信道容量、功率分配等"""
    try:
        return f"计算结果：{eval(expression)}"
    except:
        return "计算失败，请检查表达式"

def ask(question, history):
    if vectorstore is None:
        return "请先上传 PDF 论文！"
    
    agent = create_agent(
        model=llm,
        tools=[search_paper, calculate],
        system_prompt="""你是一个BD-RIS无线通信技术专家助手。
回答问题时先用search_paper工具检索论文内容，再基于检索结果回答。
如果需要计算，使用calculate工具。请用中文回答。"""
    )
    
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content

# 界面
with gr.Blocks(title="BD-RIS 智能问答系统") as demo:
    gr.Markdown("# 📡 BD-RIS 技术文档智能问答系统")
    gr.Markdown("上传BD-RIS论文，支持论文检索 + 数学计算")
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="上传 PDF 论文", file_types=[".pdf"])
            upload_btn = gr.Button("加载文档", variant="primary")
            status = gr.Textbox(label="状态", interactive=False)
            upload_btn.click(upload_pdf, inputs=pdf_input, outputs=status)
            
            gr.Markdown("### 示例问题")
            gr.Markdown("- What is the main contribution?\n- 计算 2**10 的值\n- What optimization method is used?")
        
        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                fn=ask,
                chatbot=gr.Chatbot(height=450),
                textbox=gr.Textbox(placeholder="输入问题..."),
                submit_btn="发送"
            )

if __name__ == "__main__":
    demo.launch()