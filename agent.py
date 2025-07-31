from typing import List, Any, Dict
import requests
from urllib.parse import quote

from langchain_core.tools import tool  # type: ignore
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent  # type: ignore
from langchain_core.messages import BaseMessage
from langchain_core.runnables.base import Runnable
from typing import Any

SYSTEM_PROMPT = """あなたはフレンドリーな検索AIエージェントです。
ユーザからの問い合わせに対して、日本語版 Wikipedia を検索し、関連する情報を提供します。
あなたの目的は、ユーザが求める情報を迅速かつ出展URLを明確にして提供することです。

## 利用可能なツール
- wiki_search_jp: 日本語版 Wikipedia を検索し、上位 5 件のタイトルとスニペットを返します
- wiki_summary_jp: 日本語版 Wikipedia の要約を返します。引数はページタイトルです

Markdown形式で出力してください。
"""


@tool
def wiki_search_jp(query: str) -> List[Dict[str, Any]]:
  """
  日本語版 Wikipedia で検索し、上位 5 件のタイトルとスニペットを返します。
  Args:
      query (str): 検索クエリ
  Returns:
      List[Dict[str, Any]]: 検索結果のリスト。各要素はタイトルとスニペットを含む辞書。
  """
  url = "https://ja.wikipedia.org/w/api.php"
  params: Dict[str, Any] = {
    "action": "query",
    "list": "search",
    "srsearch": query,
    "utf8": 1,
    "format": "json",
    "srprop": "snippet",
  }
  res = requests.get(url, params=params)
  data = res.json().get("query", {}).get("search", [])
  return [{"title": hit["title"], "snippet": hit["snippet"]} for hit in data[:5]]


@tool
def wiki_summary_jp(title: str) -> str:
  """
  日本語版 Wikipedia の要約を返します。
  Args:
      title (str): ページタイトル
  Returns:
      str: ページの要約
  """
  enc_title = quote(title, safe="")
  url = f"https://ja.wikipedia.org/api/rest_v1/page/summary/{enc_title}"
  headers = {"Accept-Language": "ja"}
  res = requests.get(url, headers=headers)
  data = res.json()
  return data.get("extract", "該当ページが見つかりません")


def inject_cache_control(state: Dict[str, Any]) -> Dict[str, List[Any]]:
  """
  Append cache_control to the last message in the messages list.
  """
  # state["messages"] は BaseMessage のリスト
  msgs: List[BaseMessage] = state["messages"]
  # 最後のメッセージだけに cache_control を付与する例
  new_msgs: List[Any] = []
  for i, msg in enumerate(msgs):
    print(f"Message {i}: {msg.__class__.__name__} - {msg.content.__class__.__name__}")  # type: ignore

    if i == len(msgs) - 1:
      print(f"  ## inject_cache_control")

      if isinstance(msg.content, str):  # type: ignore
        msg.content = [{"type": "text", "text": msg.content, "cache_control": {"type": "ephemeral"}}]
      elif isinstance(msg.content, list):  # type: ignore
        msg.content = [{**m, "cache_control": {"type": "ephemeral"}} if m.get("type") == "text" else m for m in msg.content]  # type: ignore
    elif i > 0:  # 0はsystemメッセージでcacheする
      if isinstance(msg.content, list):  # type: ignore
        msg.content = [{**m, "cache_control": None} if m.get("cache_control") else m for m in msg.content]  # type: ignore
      elif isinstance(msg.content, str):  # type: ignore
        msg.content = [{"type": "text", "text": msg.content}]

    new_msgs.append(msg)

  return {"llm_input_messages": new_msgs}


class WikipediaSeekAgent:
  def __init__(self) -> None:
    self.llm = ChatAnthropic(
      model_name="claude-sonnet-4-20250514",
      temperature=0.1,
      max_tokens_to_sample=2000,
      timeout=300,
      stop=[],
    )

  def _create_agent_executor(self, system_content: str) -> Runnable[Any, Any]:
    tools = [
      wiki_summary_jp,
      wiki_search_jp,
    ]

    langgraph_agent_executor: Runnable[Any, Any] = create_react_agent(  # type: ignore
      model=self.llm,
      tools=tools,
      pre_model_hook=inject_cache_control,  # register hook to inject cache_control
    )
    return langgraph_agent_executor  # type: ignore

  async def invoke(
    self,
    query: str,
  ) -> str:
    # AgentExecutorを作成
    agent_executor = self._create_agent_executor(SYSTEM_PROMPT)

    # AgentExecutorを使用して実行
    result = agent_executor.invoke(
      {
        "messages": [
          {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
          },
          {
            "role": "user",
            "content": [{"type": "text", "text": query}],
          },
        ],
      },
      {"recursion_limit": 100},
    )
    return result["messages"][-1].content
