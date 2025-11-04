from langchain_core.messages import AIMessage
import time
import json


def create_bull_researcher(llm, memory):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = investment_debate_state.get("current_response", "")
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
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""你是一位看涨分析师，主张投资该股票。
你的任务是建立一个强有力的、基于证据的案例，
强调增长潜力、竞争优势和积极的市场指标。
利用提供的研究和数据来有效解决担忧并反驳看跌论点。

重点关注：
- 增长潜力：强调公司的市场机会、收入预测和可扩展性。
- 竞争优势：强调独特产品、强大品牌或主导市场地位等因素。
- 积极指标：使用财务健康状况、行业趋势和最近的积极新闻作为证据。
- 看跌反驳点：用具体数据和合理推理批判性地分析看跌论点，全面解决担忧，并展示为什么看涨观点更有说服力。
- 参与：以对话风格呈现你的论点，直接参与看跌分析师的论点，有效辩论而非仅仅列出数据。

可用资源：
市场研究报告：{market_research_report}
社交媒体情绪报告：{sentiment_report}
最新世界事务新闻：{news_report}
公司基本面报告：{fundamentals_report}
辩论对话历史：{history}
最后的看跌论点：{current_response}
来自类似情况的反思和吸取的教训：{past_memory_str}
使用这些信息来提供一个令人信服的看涨论点，反驳看跌的担忧，并参与动态辩论，展示看涨立场的优势。你还必须处理反思并学习过去犯下的错误和教训。
"""

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
