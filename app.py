import streamlit as st
import asyncio

from agent import WikipediaSeekAgent  # Assuming the agent is defined in agent.py


async def main():
  st.set_page_config(page_title="Profile Seek Agent", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

  st.title("🤖 Profile Seek AI Agent")
  st.markdown("WebAPIのAIエージェントを使ったプロフィール検索です。")

  with st.form("profile_seek_agent_form"):
    st.subheader("検索条件")

    name = st.text_input("Name", value="勝海舟", help="検索対象の名前")

    submitted = st.form_submit_button("🔍 AI Agent検索実行")

  if submitted:
    try:
      st.subheader("検索パラメータ")

      # ProfileSeekAgentを直接使用
      agent = WikipediaSeekAgent()

      # ストリーミング検索
      st.subheader("🤖 AI Agent検索結果（リアルタイム）")

      response_placeholder = st.empty()
      full_response = ""

      # ストリーミング検索を実行
      with st.spinner("ストリーミング検索を実行中..."):
        response = await agent.invoke(name)
        st.markdown(response)

      st.success("✅ AI Agent検索完了")
      response_placeholder.markdown("### AI Agentの分析結果")
      response_placeholder.markdown(full_response)
    except Exception as e:
      st.error(f"エラーが発生しました: {str(e)}")


if __name__ == "__main__":
  asyncio.run(main())
