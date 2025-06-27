from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnableLambda
from medmcq_chatbot import format_context

# Use LangChain's wrapper (does NOT require llama-cpp-python)
llm = Ollama(model="llama3", temperature=0)

def retrieval_node(state):
    query = state["question"]
    results = state["chatbot"].vector_store.search(query)
    context = format_context(results)
    return {"context": context, "retrieval_results": results}

def generate_answer(state):
    prompt = f"""
You are a helpful medical assistant. Use the following context to answer the question.

Context:
{state['context']}

Question:
{state['question']}

Answer:"""
    answer = llm.invoke(prompt)
    return {"final_answer": answer}

# LangGraph
builder = StateGraph()
builder.add_node("retrieve", RunnableLambda(retrieval_node))
builder.add_node("generate", RunnableLambda(generate_answer))
builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()
