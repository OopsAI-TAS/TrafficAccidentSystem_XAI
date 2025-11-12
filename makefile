prepare:
	PYTHONPATH=. python3 rag/build_index.py
	PYTHONPATH=. python3 rag/query_builder.py
	PYTHONPATH=. python3 rag/query.py