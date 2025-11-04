from langchain_core.messages import AIMessage
import time
import json


def create_bear_researcher(llm, memory):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

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

        prompt = f"""你是一位看跌分析师，主张不投资该股票。
你的目标是提出一个理由充分的论点，强调风险、挑战和负面指标。
利用提供的研究和数据来突出潜在的负面影响，并有效反驳看涨论点。

重点关注：

- 风险和挑战：强调可能阻碍股票表现的因素，如市场饱和、财务不稳定或宏观经济威胁。
- 竞争劣势：强调较弱的市场地位、创新下降或来自竞争对手的威胁等脆弱性。
- 负面指标：使用来自财务数据、市场趋势或最近不利新闻的证据来支持你的立场。
- 看涨反驳点：用具体数据和合理推理批判性地分析看涨论点，暴露弱点或过度乐观的假设。
- 参与：以对话风格呈现你的论点，直接参与看涨分析师的论点，有效辩论而非仅仅列出事实。

可用资源：

市场研究报告：{market_research_report}
社交媒体情绪报告：{sentiment_report}
最新世界事务新闻：{news_report}
公司基本面报告：{fundamentals_report}
辩论对话历史：{history}
最后的看涨论点：{current_response}
来自类似情况的反思和吸取的教训：{past_memory_str}
使用这些信息来提供一个令人信服的看跌论点，反驳看涨的主张，并参与动态辩论，展示投资该股票的风险和弱点。你还必须处理反思并学习过去犯下的错误和教训。
"""

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
