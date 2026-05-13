import gradio as gr
import pymupdf
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

# 初始化模型
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = OllamaLLM(model="qwen2.5:0.5b")
vectorstore = None

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
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
    return f"✅ 成功加载 {len(doc)} 页，切分为 {len(chunks)} 个片段，可以开始提问！"

def ask(question, history):
    if vectorstore is None:
        return "请先上传 PDF 文档！"
    relevant_docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in relevant_docs])
    prompt = f"""You are an expert in wireless communications and BD-RIS technology.
Answer the question based on the following paper content:

{context}

Question: {question}
Answer:"""
    return llm.invoke(prompt)

# 界面
with gr.Blocks(title="BD-RIS 论文问答系统") as demo:
    gr.Markdown("# 📡 BD-RIS 技术文档问答系统")
    gr.Markdown("上传你的 BD-RIS 相关论文，然后用自然语言提问")
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="上传 PDF 论文", file_types=[".pdf"])
            upload_btn = gr.Button("加载文档", variant="primary")
            status = gr.Textbox(label="状态", interactive=False)
            upload_btn.click(upload_pdf, inputs=pdf_input, outputs=status)
        
        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                fn=ask,
                chatbot=gr.Chatbot(height=400),
                textbox=gr.Textbox(placeholder="输入你的问题，例如：What is BD-RIS?"),
                submit_btn="发送"
            )

if __name__ == "__main__":
    demo.launch()