pip install langgraph langchain-aws langchain-core


from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# 1. Connect Bedrock LLM
llm = ChatBedrock(
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1",
    model_kwargs={
        "max_tokens": 1000,
        "temperature": 0
    }
)

# 2. Define State
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# 3. Define Node
def call_llm(state: AgentState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# 4. Build Graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.set_entry_point("llm")
graph.add_edge("llm", END)

app = graph.compile()

# 5. Run
result = app.invoke({
    "messages": [HumanMessage(content="What is pharmacovigilance?")]
})

print(result["messages"][-1].content)
