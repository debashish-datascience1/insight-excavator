.PHONY: run ui test generate-data clean

run:
	uvicorn app.main:app --reload

ui:
	streamlit run ui/streamlit_app.py

test:
	pytest tests/ -v

generate-data:
	python data/sample/generate_sample.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	rm -rf .pytest_cache .llm_cache
