from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_news, get_global_news
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        tools = [
            get_news,
            get_global_news,
        ]

        system_message = (
            "你是一位新闻研究员，负责分析过去一周的最新新闻和趋势。"
            "请编写一份与交易和宏观经济相关的世界当前状况的综合报告。"
            "使用可用工具：get_news(query, start_date, end_date) "
            "用于公司特定或定向新闻搜索，"
            "get_global_news(curr_date, look_back_days, limit) "
            "用于更广泛的宏观经济新闻。"
            "不要简单地说明趋势是混合的，提供详细和细粒度的分析和见解，"
            "可能有助于交易员做出决策。"
            + """ 确保在报告末尾附加一个Markdown表格，以组织报告中的关键点，使其有条理且易于阅读。"""
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
                    "供参考，当前日期是 {current_date}。我们正在查看的公司是 {ticker}",
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
            "news_report": report,
        }

    return news_analyst_node
