from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_news
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = [
            get_news,
        ]

        system_message = (
            "你是一位社交媒体和公司特定新闻研究员/分析师，"
            "负责分析过去一周特定公司的社交媒体帖子、最新公司新闻和公众情绪。"
            "你会得到一个公司名称，你的目标是编写一份全面的长篇报告，"
            "详细说明你的分析、见解以及对交易员和投资者的影响，"
            "包括查看社交媒体和人们对该公司的评论、"
            "分析人们每天对公司的情绪数据，以及查看最近的公司新闻。"
            "使用 get_news(query, start_date, end_date) 工具搜索公司特定新闻和社交媒体讨论。"
            "尝试查看所有可能的来源，从社交媒体到情绪到新闻。"
            "不要简单地说明趋势是混合的，提供详细和细粒度的分析和见解，"
            "可能有助于交易员做出决策。"
            + """ 确保在报告末尾附加一个Markdown表格，以组织报告中的关键点，使其有条理且易于阅读。""",
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一位有用的AI助手，与其他助手协作。"
                    " 使用提供的工具来推进回答问题的过程。"
                    " 如果你无法完全回答，没关系；其他具有不同工具的助手"
                    " 会在你离开的地方继续提供帮助。执行你能做的以推进进度。"
                    " 如果你或任何其他助手有最终交易建议：**买入/持有/卖出**或可交付成果，"
                    " 请在回复前加上最终交易建议：**买入/持有/卖出**，以便团队知道停止。"
                    " 你可以访问以下工具：{tool_names}。\n{system_message}"
                    "供参考，当前日期是 {current_date}。我们想要分析的当前公司是 {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "sentiment_report": report,
        }

    return social_media_analyst_node
