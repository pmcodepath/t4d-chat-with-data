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
    st.subheader("Step 1: Upload and Analyze Data")

    uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            df = pd.read_csv(file)
            st.session_state.dataframes[file.name] = df
            st.success(f"Uploaded: {file.name}")

        if st.button("Analyze Data"):
            for name, df in st.session_state.dataframes.items():
                st.write(f"Analysis for {name}:")
                st.write(f"Shape: {df.shape}")
                st.write("Columns:")
                st.write(df.columns.tolist())
                st.write("Preview:")
                st.write(df.head())
                st.write("---")

        if st.button("Proceed to Data Cleaning"):
            st.session_state.step = 2

def step_2_clean_data():
    st.subheader("Step 2: Clean Data")

    llm = OpenAI(temperature=0)
    
    for name, df in st.session_state.dataframes.items():
        st.write(f"Cleaning recommendations for {name}:")
        
        # Create a summary of the dataframe
        summary = f"Dataframe '{name}' summary:\n"
        summary += f"- Shape: {df.shape}\n"
        summary += f"- Columns: {', '.join(df.columns)}\n"
        summary += "- Data types:\n"
        for col, dtype in df.dtypes.items():
            summary += f"  - {col}: {dtype}\n"
        summary += "- Sample data (first 5 rows):\n"
        summary += df.head().to_string()

        # Split the summary into smaller chunks
        chunk_size = 1500  # Reduced chunk size
        chunks = textwrap.wrap(summary, chunk_size)

        cleaning_recommendations = []
        with st.spinner("Analyzing data and generating recommendations..."):
            for i, chunk in enumerate(chunks):
                chunk_result = analyze_chunk(llm, df, chunk)
                cleaning_recommendations.append(chunk_result)

        # Combine all recommendations
        full_recommendations = "\n".join(cleaning_recommendations)
        st.write(full_recommendations)
        
        # Create checkboxes for cleaning operations
        cleaning_ops = [op.strip() for op in full_recommendations.split('\n') if op.strip()]
        st.session_state.cleaning_operations[name] = []
        for op in cleaning_ops:
            if st.checkbox(op, key=f"{name}_{op}"):
                st.session_state.cleaning_operations[name].append(op)

    if st.button("Apply Cleaning and Proceed to Chat"):
        for name, ops in st.session_state.cleaning_operations.items():
            df = st.session_state.dataframes[name]
            for op in ops:
                # Here you would implement the actual cleaning operations
                # For now, we'll just print what would be done
                st.write(f"Applying to {name}: {op}")
        
        st.session_state.step = 3
        st.success("Cleaning operations applied. Proceeding to chat interface.")
        st.button("Go to Chat Interface")

    if st.button("Back to Data Upload"):
        st.session_state.step = 1
        st.experimental_rerun()

def step_3_chat_with_data():
    st.subheader("Step 3: Chat with your data")

    user_input = st.text_input("Ask a question about your data:")
    if user_input:
        response = process_user_input(user_input)
        st.session_state.chat_history.append(("User", user_input))
        st.session_state.chat_history.append(("AI", response))

    for role, message in st.session_state.chat_history:
        if role == "User":
            st.text_area("You:", value=message, height=50, disabled=True)
        else:
            st.text_area("AI:", value=message, height=100, disabled=True)

def process_user_input(user_input):
    llm = OpenAI(temperature=0)
    combined_df = pd.concat([df.assign(source=name) for name, df in st.session_state.dataframes.items()], ignore_index=True)

    df_summary = "Available data:\n"
    for name, df in st.session_state.dataframes.items():
        df_summary += f"- {name}: {len(df)} rows, {len(df.columns)} columns\n"
        df_summary += f"  Columns: {', '.join(df.columns)}\n\n"

    agent = create_pandas_dataframe_agent(
        llm,
        combined_df,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True
    )

    full_input = f"{df_summary}\nThe data from all files has been combined into a single DataFrame with an additional 'source' column indicating the original file.\n\nUser question: {user_input}"

    response = agent.run(full_input)
    return response

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
