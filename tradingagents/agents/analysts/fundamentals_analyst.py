from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement, get_insider_sentiment, get_insider_transactions
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        system_message = (
            "你是一位研究员，负责分析过去一周关于公司的基本面信息。请编写一份关于公司基本面信息的综合报告，如财务文件、公司概况、基本公司财务数据和公司财务历史，以获得公司基本面信息的完整视图，为交易员提供信息。确保包含尽可能多的细节。不要简单地说明趋势是混合的，提供详细和细粒度的分析和见解，可能有助于交易员做出决策。"
            + " 确保在报告末尾附加一个Markdown表格，以组织报告中的关键点，使其有条理且易于阅读。"
            + " 使用可用工具：`get_fundamentals` 用于综合公司分析，`get_balance_sheet`、`get_cashflow` 和 `get_income_statement` 用于特定财务报表。",
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
                    "供参考，当前日期是 {current_date}。我们要查看的公司是 {ticker}",
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
