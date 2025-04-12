# 17630-StudentGuideAi

### Create .env with the following variable:
1. ``OPENAI_API_KEY=`` 
2. ``OPENAI_ORG=``
3. ``OPENAI_PROJECT_ID=``

### How to Run the Project?
- Automated Setup:
1. Run the script using ``./setup_and_run.sh``

- Manual Setup:
1. Create a local virtual environment using ``python3 -m venv prompt-project``
2. Activate the environment using `` source prompt-project/bin/activate``
3. Install the necessary dependencies using ``pip install -q -r requirements.txt``
4. Build the RAG using ``rag.py``
5. Run the project in the localhost using ``python main.py``
