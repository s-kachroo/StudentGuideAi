# 17630-StudentGuideAi

### Problem Statement

New and current Carnegie Mellon University (CMU) students face challenges navigating the complex academic ecosystem. An overwhelming volume of academic policies, like understanding cross-registration, involves numerous rules and forms​. Information is spread across multiple data sources in university policy PDFs, departmental FAQs, and college handbooks. The generic Q&A chatbots fall short for CMU-specific queries, and the static FAQ pages lack interactive guidance, while general-purpose large language models aren’t tailored to CMU’s latest policies and may hallucinate university-specific rules.
For example, a generic chatbot might confidently give the wrong course add deadline or an outdated graduation requirement. No centralized tool guided students from orientation to graduation on CMU-specific academic matters. These gaps motivated the creation of “StudentGuideAi,” an AI-powered assistant that provides reliable answers and streamlines the student experience.

### Create .env with the following variable:
1. ``OPENAI_API_KEY=`` 
2. ``OPENAI_ORG=``
3. ``OPENAI_PROJECT_ID=``

### Experimentation and RAG Workflow?
- We have python notebooks in the [folder](./experiments/).
- [WIP] We're constantly improving our experimentations.

### How to Run the Project?

- Automated Setup:
1. Run the script using ``./setup_and_run.sh``

- Manual Setup:
1. Create a local virtual environment using ``python3 -m venv prompt-project``
2. Activate the environment using `` source prompt-project/bin/activate``
3. Install the necessary dependencies using ``pip install -q -r requirements.txt``
4. Build the RAG using ``rag.py``
5. Run the project on localhost using ``python main.py``

### Documentation
- We have drafted a slideshow [(Click here)](https://docs.google.com/presentation/d/1xJBs_CK4z5W1UstiK5AJD_dHDiSltdr_) and a report [(Click here)](https://docs.google.com/document/d/1qRhkiyXRzB64xOAa9-rwfxL7Ev16P2OIkWXxQu7__P4)  about the project.
