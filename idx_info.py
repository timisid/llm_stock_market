import os
import json
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from datetime import datetime
import pandas as pd
import altair as alt



# import library for conversational memory
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Import streamlit for app dev
import streamlit as st


load_dotenv() # load your .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SECTORS_API_KEY = os.getenv("SECTORS_API_KEY")

def retrieve_from_endpoint(url: str) -> dict: 
    headers = {"Authorization": SECTORS_API_KEY}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

    return json.dumps(data)

@tool
def get_company_overview(stock: str, section: str) -> str:
    """
    Get company by : overview
    Get company listed
    Calculated the company listed
    Harga Saham Terakhir
    get company address, get description, get history
    """
    url = f"https://api.sectors.app/v1/company/report/{stock}/?sections=overview"

    return retrieve_from_endpoint(url)

@tool
def get_company_management(stock: str, section: str) -> str:
    """
    Get company by : management
    Get company listed
    stakeholder of the company
    """
    url = f"https://api.sectors.app/v1/company/report/{stock}/?sections=management"

    return retrieve_from_endpoint(url)


@tool
def get_company_peers(stock: str, section: str) -> str:
    """
    Get company by : peers
    asset value, pe, pb
    """
    url = f"https://api.sectors.app/v1/company/report/{stock}/?sections=peers"

    return retrieve_from_endpoint(url)

@tool
def get_company_dividend(stock: str, section: str) -> str:
    """
    Get company by : dividend
    """
    url = f"https://api.sectors.app/v1/company/report/{stock}/?sections=dividend"

    return retrieve_from_endpoint(url)

@tool
def get_company_by_index(indeks: str) -> str:
    """
    Get sector
    get company subsector
    """
    url = f"https://api.sectors.app/v1/index/{indeks}/"

    return retrieve_from_endpoint(url)

@tool
def get_company_by_subsector(subsector: str) -> str:
    """
    Get sector name groped in subsector
    """
    url = f"https://api.sectors.app/v1/companies/?sub_sector={subsector}"

    return retrieve_from_endpoint(url)


@tool
def get_top_company(start_date: str, end_date: str, sub_sector: int = 5) -> str:
    """
    Get sector
    get company subsector
    """
    url = f"https://api.sectors.app/v1/most-traded/?start={start_date}&end={end_date}&n_stock=5&sub_sector={sub_sector}"

    return retrieve_from_endpoint(url)


 
tools = [
    get_company_overview,
    get_company_peers, 
    get_company_management, 
    get_company_dividend,
    get_company_by_index, 
    get_company_by_subsector,
    get_top_company
]

llm = ChatGroq(
    temperature=0, 
    model_name="llama3-groq-70b-8192-tool-use-preview", 
    groq_api_key= GROQ_API_KEY,
)

#streamlit section 

st.image("stock.jpg")
st.title(":page_facing_up: :green[IDX INFO] ")
st.markdown("Anything in general info for Indonesian Stock Market (IDX) by using sectors app API")
st.markdown(":orange[Tools that included are subsector, company report overview, management, peers, dividen and stock price] ")

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st.write("🧠 thinking...")
        st_callback = StreamlitCallbackHandler(st.container())
        promptt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""
                    you are "Ismi Asistant", you only process stocks data based in indonesia. 
    
                    Answer the following queries, being as factual and analytical as you can 
                    if possible strictly numbers that you hit from API tools. 

                    If you need the start and end dates but they are not provided, try this step: 
                    1. get 'date' == start_date == end_date

                    if data was empty try this step: 
                    1. get the next date (start_date+1) to proceed in invoke
                    2. if still empty, get the next another date (start_date+2) to proceed in invoke
                    3. return the result by writing the invoke result. also write the start_date that proceed

                    If the volume was about a single day, the start and end parameter should be the same.  
                    if there is a question using over several days, then sum it based on the days given

                    today is {datetime.now().strftime("%Y-%m-%d")} use this as time context. 
                    strictly always answer in markdown table if you can or neat table, also you can answer in prettify json if you can.

                    """
                ), 
                ("human", prompt), 
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        agent=create_tool_calling_agent(llm, tools, promptt)
        agent_executor= AgentExecutor(agent=agent, tools=tools, verbose=True)
        result = agent_executor.invoke({"input":prompt}) 

        st.session_state['key'] = 'value'
        st.write(result["output"])



        

