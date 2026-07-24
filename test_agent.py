from vector_store import HybridVectorStore
from agent_graph import AgenticRAGGraph

def test_vector_store():
    store = HybridVectorStore()
    results = store.hybrid_search("Continuous Improvement Agent GSK", top_k=2)
    assert len(results) > 0
    assert "GSK" in results[0]["content"] or "CIA" in results[0]["content"]

def test_agent_graph_execution():
    graph = AgenticRAGGraph()
    res = graph.execute_query("What are GHG emissions for aluminium smelting?")
    assert "final_answer" in res
    assert len(res["trajectory"]) == 4
    assert res["ragas_scores"]["faithfulness"] > 0.80
    assert len(res["citations"]) > 0

if __name__ == "__main__":
    test_vector_store()
    test_agent_graph_execution()
    print("All Agentic RAG pipeline unit tests passed successfully!")
