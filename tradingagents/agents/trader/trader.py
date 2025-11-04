import functools
import time
import json


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = (
            f"{market_research_report}\n\n"
            f"{sentiment_report}\n\n"
            f"{news_report}\n\n"
            f"{fundamentals_report}"
        )
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "未找到过去的记忆。"

        context = {
            "role": "user",
            "content": (
                f"基于分析师团队的全面分析，这是为 {company_name} 量身定制的投资计划。"
                f"该计划融合了当前技术市场趋势、宏观经济指标和社交媒体情绪的见解。"
                f"使用此计划作为评估你下一个交易决策的基础。\n\n"
                f"拟议投资计划：{investment_plan}\n\n"
                f"利用这些见解做出明智和战略性的决策。"
            ),
        }

        messages = [
            {
                "role": "system",
                "content": (
                    f"""你是一位分析市场数据以做出投资决策的交易代理。
基于你的分析，提供具体的买入、卖出或持有建议。
以明确的决定结束，并始终以"最终交易建议：**买入/持有/卖出**"
结束你的回复，以确认你的建议。
不要忘记利用过去决策的教训来从错误中学习。
以下是你交易过的类似情况的反思和吸取的教训：{past_memory_str}"""
                ),
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
