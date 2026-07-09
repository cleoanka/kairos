"""The Microstructure Analyst — System-1 perception, spoken into System-2.

A new member of the analyst team whose tools are backed by the causal
perception bus (LOB-Core), not a data vendor. It gives the bull/bear debate a
grounding the original TradingAgents lineup lacks: *what the order book is
actually doing right now*, delivered under a hard no-look-ahead guarantee.

Requires the ``[reasoning]`` extra. Imported only by the graph wiring.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from kairos.reasoning.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)

from .microstructure_tools import get_microstructure_regime, get_order_flow_state


def create_microstructure_analyst(llm):
    def microstructure_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        tools = [get_microstructure_regime, get_order_flow_state]

        system_message = (
            """You are the Microstructure Analyst, the team's System-1 sense of the market. Unlike the other analysts, you do not read price history, fundamentals, or news — you read the **limit order book itself**: the live regime, order-flow imbalance, resting-depth pressure, and liquidity toxicity, delivered as a strictly point-in-time percept.

Your job: call get_microstructure_regime for this symbol and date to get the current System-1 read, and get_order_flow_state to judge whether that regime is stable or flickering over the recent causal window. Then write a concise report on what the microstructure implies for a trade *right now*.

Interpret the signals like a market maker, not a chart technician:
- RANGE  → balanced two-sided liquidity; mean-reversion and spread capture are viable.
- TREND  → one-sided aggressive flow is consuming the book; do not fade it; favor reduce-only / directional continuation.
- TOXIC  → displayed liquidity is phantom (spoofing / cancels dominate). This is a STAND-ASIDE signal. Say so explicitly and warn the trader that executing here is hazardous regardless of the fundamental view.
- Order-flow imbalance and depth imbalance give a BULL/BEAR lean; treat weak/near-zero readings as no edge.

Two hard rules:
1. Every number you cite must come from a tool call. If the tools report perception is unavailable at this date, say the microstructure read is unavailable and defer to the other analysts — never invent a regime or a level.
2. You are describing the *state of liquidity*, not forecasting price. Do not claim a price target."""
            + """ Append a short Markdown table summarizing regime, direction, order-flow imbalance, depth imbalance, and toxicity."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([t.name for t in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {"messages": [result], "microstructure_report": report}

    return microstructure_analyst_node
