import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_community.llms import OpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import time

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'cleaning_operations' not in st.session_state:
    st.session_state.cleaning_operations = {}

def main():
    st.title("Data Analysis Chat App")

    if st.session_state.step == 1:
        step_1_upload_and_analyze()
    elif st.session_state.step == 2:
        step_2_clean_data()
    elif st.session_state.step == 3:
        step_3_chat_with_data()

def step_1_upload_and_analyze():
    st.header("Step 1: Upload and Analyze Data")

    uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True, type="csv")

    if uploaded_files:
        st.session_state.uploaded_files = {}
        for uploaded_file in uploaded_files:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_files[uploaded_file.name] = df
            
            st.subheader(f"Preview of {uploaded_file.name}")
            st.write(df.head())
            
            st.subheader(f"Summary of {uploaded_file.name}")
            st.write(df.describe())

        st.write("Debug: Uploaded files in session state:")
        st.write({k: v.shape for k, v in st.session_state.uploaded_files.items()})

        if st.button("Proceed to Data Cleaning"):
            st.session_state.step = 2
            st.rerun()
    else:
        st.info("Please upload one or more CSV files to proceed.")

    # Debug information
    st.subheader("Debug Information")
    st.write("Session State Contents:")
    st.write(st.session_state)

def step_2_clean_data():
    st.header("Step 2: Clean Data")

    st.write("Debug: Session State in step 2:")
    st.write(st.session_state)

    if 'uploaded_files' not in st.session_state or not st.session_state.uploaded_files:
        st.warning("No files uploaded. Please go back to Step 1 and upload files.")
        if st.button("Back to Data Upload"):
            st.session_state.step = 1
            st.rerun()
        return

    cleaning_options = {}
    for file_name, df in st.session_state.uploaded_files.items():
        st.subheader(f"Cleaning options for {file_name}")
        
        file_options = []
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            file_options.append(f"Remove outliers from numeric columns in {file_name}")
        
        # Text columns
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            file_options.append(f"Remove leading/trailing whitespace from text columns in {file_name}")
            file_options.append(f"Convert text to lowercase in {file_name}")
        
        # Date columns
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            file_options.append(f"Standardize date format in {file_name}")
        
        # Missing values
        if df.isnull().any().any():
            file_options.append(f"Handle missing values in {file_name}")
        
        # Duplicate rows
        if df.duplicated().any():
            file_options.append(f"Remove duplicate rows from {file_name}")
        
        cleaning_options[file_name] = st.multiselect(
            f"Select cleaning operations for {file_name}:",
            options=file_options
        )

    if st.button("Clean Data"):
        for file_name, options in cleaning_options.items():
            df = st.session_state.uploaded_files[file_name]
            
            for option in options:
                if "Remove outliers" in option:
                    # Implement outlier removal
                    for col in df.select_dtypes(include=['int64', 'float64']).columns:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
                
                elif "Remove leading/trailing whitespace" in option:
                    # Remove whitespace from text columns
                    for col in df.select_dtypes(include=['object']).columns:
                        df[col] = df[col].str.strip()
                
                elif "Convert text to lowercase" in option:
                    # Convert text to lowercase
                    for col in df.select_dtypes(include=['object']).columns:
                        df[col] = df[col].str.lower()
                
                elif "Standardize date format" in option:
                    # Standardize date format
                    for col in df.select_dtypes(include=['datetime64']).columns:
                        df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
                
                elif "Handle missing values" in option:
                    # Handle missing values (using mean for numeric, mode for categorical)
                    for col in df.columns:
                        if df[col].dtype in ['int64', 'float64']:
                            df[col].fillna(df[col].mean(), inplace=True)
                        elif df[col].dtype == 'object':
                            mode_value = df[col].mode()
                            if not mode_value.empty:
                                df[col].fillna(mode_value[0], inplace=True)
                            else:
                                df[col].fillna('Unknown', inplace=True)
                
                elif "Remove duplicate rows" in option:
                    # Remove duplicate rows
                    df.drop_duplicates(inplace=True)
            
            st.session_state.uploaded_files[file_name] = df  # Store the cleaned dataframe back
            st.session_state.dataframes[file_name] = df

        st.success("Data cleaning completed!")
        st.write("Debug: Cleaned data in session state:")
        st.write({k: v.shape for k, v in st.session_state.uploaded_files.items()})
        st.session_state.step = 3
        st.rerun()

    if st.button("Back to Data Upload"):
        st.session_state.step = 1
        st.rerun()

def step_3_chat_with_data():
    st.header("Step 3: Chat with Data")

    st.write("Debug: Session State in step 3:")
    st.write(st.session_state)

    if 'uploaded_files' not in st.session_state or not st.session_state.uploaded_files:
        st.warning("No cleaned data available. Please go back to Step 2 and clean the data.")
        if st.button("Back to Data Cleaning"):
            st.session_state.step = 2
            st.rerun()
        return

    st.write("Debug: Data in session state:")
    st.write({k: v.shape for k, v in st.session_state.uploaded_files.items()})

    if prompt := st.chat_input("Ask a question about the data"):
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            response = process_user_input(prompt)
            st.write(response)

    if st.button("Back to Data Cleaning"):
        st.session_state.step = 2
        st.rerun()

import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamlitCallbackHandler

def process_user_input(user_input):
    st.write("Debug: Entering process_user_input function")
    st.write(f"Debug: Session state keys: {st.session_state.keys()}")

    if 'uploaded_files' not in st.session_state or not st.session_state.uploaded_files:
        st.error("No data available. Please upload and clean data first.")
        return "No data available to process the query."

    try:
        combined_df = pd.concat([df for df in st.session_state.uploaded_files.values()], ignore_index=True)
        st.write("Debug: Combined dataframe shape:", combined_df.shape)
        st.write("Debug: Combined dataframe head:")
        st.write(combined_df.head())
    except Exception as e:
        st.error(f"Error combining dataframes: {str(e)}")
        st.write("Contents of st.session_state.uploaded_files:")
        st.write(st.session_state.uploaded_files)
        return "Error processing data. Please check the debug information."

    # Check if OpenAI API key is set
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
        return "Error: OpenAI API key is not set."

    try:
        # Create an OpenAI language model
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

        # Create a prompt template
        prompt = PromptTemplate(
            input_variables=["query", "data_description"],
            template="""
            You are an AI assistant tasked with analyzing data and answering questions about it.
            
            Data Description:
            {data_description}
            
            User Query: {query}
            
            Please provide a detailed and accurate answer based on the given data description.
            If the query cannot be answered with the available information, please state so clearly.
            """
        )

        # Create an LLM chain
        chain = LLMChain(llm=llm, prompt=prompt)

        # Generate a data description
        data_description = f"The dataframe has {combined_df.shape[0]} rows and {combined_df.shape[1]} columns. "
        data_description += f"The columns are: {', '.join(combined_df.columns)}. "
        data_description += "Some basic statistics of the numerical columns: \n"
        data_description += combined_df.describe().to_string()

        # Add some sample data
        data_description += "\n\nHere are the first few rows of the data:\n"
        data_description += combined_df.head().to_string()

        # Run the chain with a Streamlit callback
        with st.spinner("Analyzing your query..."):
            with st.container():
                st_callback = StreamlitCallbackHandler(st.container())
                response = chain.run(query=user_input, data_description=data_description, callbacks=[st_callback])
        
        return response
    except Exception as e:
        st.error(f"An error occurred while processing the query: {str(e)}")
        return f"Error: {str(e)}"

def analyze_chunk(llm, df, chunk, timeout=30):
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True
    )
    
    prompt = f"Analyze this part of the dataframe summary and suggest up to 3 specific cleaning operations. Focus on identifying missing values, outliers, and inconsistent data formats.\n\n{chunk}"
    
    try:
        with ThreadPoolExecutor() as executor:
            future = executor.submit(agent.run, prompt)
            return future.result(timeout=timeout)
    except Exception as e:
        return f"Analysis timed out or encountered an error: {str(e)}"

if __name__ == "__main__":
    main()
