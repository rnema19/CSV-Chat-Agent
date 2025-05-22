
Gen AI CSV File Chatbot using Langchain



## Installation

Install project with python. Use Jupyter Notebook.

Download the required libraries by:

```bash
pip install -r requirements.txt
```
set up .env file and in .env file write your API keys generated from GROQ, OPEN AI and Langsmith API keys or any API if needed for other files like PDF reader
(not needed for CSV chatbot)

Activate .venv file by (to setup a virtual environment):

```bash
.venv\Scripts\Activate.ps1

```

Download Ollama in your PC.
After downloading run 

```bash
ollama run qwen3:1.7b
```
in your powershell or Command Prompt


Run your app on streamlit 
``` bash
streamlit run appCSV.py
```
## Documentation
Use Ananconda or Jupyter Notebooks for running environment.

[Jupyter Notebook](https://jupyter.org/)

[Ollama - qwen3](https://ollama.com/library/qwen3)

[Langchain](https://python.langchain.com/v0.2/docs/introduction/)

[Langsmith](https://docs.smith.langchain.com/)

[CSV-Agent](https://python.langchain.com/api_reference/experimental/agents/langchain_experimental.agents.agent_toolkits.csv.base.create_csv_agent.html)






## Acknowledgements

 - [Building a CSV assistant]https://blog.mlq.ai/csv-assistant-langchain/)
 



