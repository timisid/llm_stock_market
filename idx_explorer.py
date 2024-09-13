import os
import json
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from datetime import date, datetime, timedelta
import pandas as pd

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
def get_company_overview(stock: str) -> str:
    """
    Get company overview
    Get company listed
    Calculated the company listed
    Harga Saham Terakhir
    get company address, get description, get history
    """
    url = f"https://api.sectors.app/v1/company/report/{stock}/?sections=overview"

    return retrieve_from_endpoint(url)

@tool
def get_trx_volume_raw (start_date: str, end_date: str, top_n: int = 5) -> str:
    """
    top 5 companies by transaction volume on the first of this month
    """

    url = f"https://api.sectors.app/v1/most-traded/?start={start_date}&end={end_date}&n_stock={top_n}"
    return retrieve_from_endpoint(url)

@tool
def get_top_companies_tx_calculated (start_date: str, end_date: str, top_n: int = 5) -> str:
    """
    get top  5 or top 3 companies by transaction volume in several days group by symbol and sum by volume
    top most traded
    
    """
    url = f"https://api.sectors.app/v1/most-traded/?start={start_date}&end={end_date}&n_stock={top_n}"

    x = retrieve_from_endpoint(url)
    df = pd.DataFrame()
    x = json.loads(x)
    for date, records in x.items():
        temdf = pd.DataFrame(records)
        temdf['date'] = date
        df = pd.concat([df, temdf])
    
    top_companies = df.groupby(['symbol', 'company_name'])['volume'].sum().reset_index()
    top_companies = top_companies.sort_values(by=['volume'], ascending=False).head(top_n).reset_index()

    # return retrieve_from_endpoint(url)
    return top_companies

@tool
def get_daily_tx (stock: str, start_date: str, end_date: str) -> str:
    """
    Get daily transaction for a stock
    Get latest price on every stock on last day of the year
    
    """
    url = f"https://api.sectors.app/v1/daily/{stock}/?start={start_date}&end={end_date}"

    return retrieve_from_endpoint(url)

@tool
def get_performance_since_ipo(stock: str) -> str:
    """
    Get stock performance since IPO listing for a given stock symbol.
    write it from 7 days, 30 days, 90 days to 365 days
    """
    url = f"https://api.sectors.app/v1/listing-performance/{stock}/"


    return retrieve_from_endpoint(url)


tools = [
    get_company_overview,
    get_trx_volume_raw,
    get_top_companies_tx_calculated,
    get_daily_tx, 
    get_performance_since_ipo
]


llm = ChatGroq(
    temperature=0, 
    model_name="llama3-groq-70b-8192-tool-use-preview", 
    groq_api_key= GROQ_API_KEY,
)


st.image("stock.jpg")
st.title(":face_with_monocle: :green[IDX Explorer] ")
st.markdown("Help you to emphasize your knowledge for Indonesian Stock Market (IDX) by using sectors app API")
st.markdown(":orange[Tools that included are company overview, most traded by trx volume, daily stock trx, ipo performance] ")

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

                    Answer the following queries, being as factual and analytical as you can. 
                    If you need the start and end dates but they are not 
                    explicitly provided (example contains: least, terakhir), get last date of today in current date tools. 
                    Whenever you return a list of names, return also the corresponding values for each name. 

                    this is important : strictly  take out column named "rank" in markdown table, prohibited to show
            
                    If the volume was about a single day, the start and end parameter should be the same.  
                    if there is a question using over several days, then sum it based on the days given

                    today is {datetime.now().strftime("%Y-%m-%d")} use this as time context. 

                    for def get_trx_volume_raw : detect the day of the start_date.
                    if the start_date is sunday, then add the day by one day (start_date+1), 
                    if the start_date is saturday, then add the day by one day (start_date+2)
                    example: 1st day of the month is sunday,then take 2nd day of the month to proceed as start_date.
                    or if the 1st day of the month is saturday, then take the 3rd day of the month to proceed as start_date

                    always answer in markdown table if you can or neat table, also you can answer in prettify json if you can.

                    strictly if there are day range in result, sort it from the first day of the month to the last day of the month

                    if data invoke was empty try this step: 
                    1. get the next date (start_date+1) to proceed in invoke
                    2. if still empty, get the next another date (start_date+2) to proceed in invoke
                    3. return the result by writing the invoke result. also write the start_date that proceed


                    """
                ), 
                ("human", prompt), 
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        agent=create_tool_calling_agent(llm, tools, promptt)
        agent_executor= AgentExecutor(agent=agent, tools=tools, verbose=True)
        result = agent_executor.invoke({"input":prompt})        
        # response = agent.run(prompt, callbacks=[st_callback])
        st.write(result["output"])

# query_4 = "Berapa harga saham BBRI tahun 2024?"
# query_5 = "berapa nilai market cap BMRI terakhir?"

# # # queries = [query_1, query_2, query_3, query_4, query_5]
# queries = [query_4, query_5]


# for query in queries:
#     print("Question:", query)
#     result = agent_executor.invoke({"input": query})
#     print("Answer:", "\n", result["output"], "\n\n======\n\n")
