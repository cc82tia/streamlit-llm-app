import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()  # ← .envファイルの読み込み
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, max_tokens=1000, api_key=os.getenv("OPENAI_API_KEY"))

st.title(" 生成AI専門家アシスタント")

st.write(
    """#####
このアプリでは、2人のAI専門家に質問を投げかけ、生成AIの活用方法についてアドバイスを受けることができます。質問内容に応じて、適切な専門家を選んでください。
---
###### 🔧 使い方
1. **専門家の種類を選択**  
   - 「AIエンジニア」：PythonやLangChainの使い方、RAGや埋め込みベクトルに関する質問向け  
   - 「プロンプトエンジニア」：プロンプトの書き方、業務活用、ユースケース設計などの相談向け  
2. **質問を入力**  
   - フォームに自由な質問文を入力してください（例：「RAGって何ですか？」）
3. **回答を確認**  
   - 数秒後に、専門家からの回答が画面に表示されます。
---
###### 💬 ご利用例
- 「RAGシステムをStreamlitで動かすにはどうすればいいですか？」
- 「営業支援に向いたプロンプトの構造を教えてください」
---
📌 *本アプリは学習目的で開発されたものであり、生成された回答は参考情報としてご利用ください。*
""")

selected_expert = st.radio("専門家を選択してください",  ["AIエンジニア", "プロンプトエンジニア"])

user_question = st.text_input("質問を入力してください", placeholder="例：営業支援に向いたプロンプトの構造を教えてください")

def get_expert_response(selected_expert, user_question):
    """radioボタンで選択された専門家に応じて、適切なメッセージを返す関数"""
    if selected_expert == "AIエンジニア":
        systemmessage = """You are an AI engineer specializing in Python and LangChain. 
                          You help users understand how to implement 
                          RAG (Retrieval-Augmented Generation) systems, 
                          use embedding vectors, and leverage other advanced techniques in AI development.
                          Please respond in Japanese."""
        
    else:
        systemmessage = """You are a prompt engineer and AI consultant who specializes in helping beginners design effective prompts and business use cases for generative AI. Your goal is to guide users in crafting prompts that achieve specific goals, such as automating tasks, retrieving relevant knowledge, or summarizing content. You are friendly and business-focused, and you always help clarify the purpose behind each prompt to ensure value-driven AI usage.
You provide practical examples and actionable advice to help users understand how to structure their prompts for maximum effectiveness. Always encourage users to think about the end goal of their prompts and how they can best achieve it with generative AI.
Please respond in Japanese."""
    
    messages = [
        SystemMessage(content=systemmessage),
        HumanMessage(content=user_question)
    ]
    return llm(messages)

if st.button("質問を送信"):
    if user_question:
        st.divider()
        try:
                result = get_expert_response(selected_expert, user_question)
                st.write(f"**{selected_expert}の回答：**")
                st.write(result.content)
        except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
    else:
        st.warning("質問を入力してください。")
