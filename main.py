from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import init_chat_model
import pandas as pd
import base64
import os
import uuid
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Load the Titanic dataset
df = pd.read_csv('titanic.csv')

# Initialize the model
model = init_chat_model("llama3-70b-8192", model_provider="groq")

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
async def home():
    return {"message": "Welcome to the Titanic dataset assistant!"}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    question = request.question
    plot_filename = f"temp_plot_{uuid.uuid4().hex}.png"

    # Custom prompt with dynamic plot filename
    custom_prefix = f"""
You are a helpful data science assistant analyzing the Titanic dataset in a pandas DataFrame 'df'. Follow these rules:

1. For non-visualization questions, generate code that prints the answer.
2. For visualization requests:
   - Use matplotlib.pyplot to create the plot.
   - Save the plot to '{plot_filename}'.
   - Close the plot with plt.close().
   - Print a description of the visualization.

Examples:
- User asks for average fare: print(df['Fare'].mean())
- User asks for a histogram: 
  import matplotlib.pyplot as plt
  df['Age'].hist()
  plt.savefig('{plot_filename}')
  plt.close()
  print("Histogram generated showing passenger age distribution.")
"""

    agent = create_pandas_dataframe_agent(
        model,
        df,
        verbose=True,
        prefix=custom_prefix,
        allow_dangerous_code=True,
    )

    try:
        text_answer = agent.invoke({"input": question})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Handle plot generation
    plot_base64 = None
    if os.path.exists(plot_filename):
        with open(plot_filename, "rb") as f:
            plot_base64 = base64.b64encode(f.read()).decode("utf-8")
        os.remove(plot_filename)

    return {"text": text_answer.get("output"), "plot": plot_base64}
