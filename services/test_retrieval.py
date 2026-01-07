from retrieval_service import search


results = search(
    query_text="¿Cómo funciona el picking en un WMS?",
    domain="wms",
    language="es",
    top_k=5
)

for r in results:
    print(f"• ({r['similarity']:.4f}) {r['content']}")
