import os
import chainlit as cl

from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")  # Change this to your actual API key environment variable

provider = AsyncOpenAI(
    api_key=api_key,
    base_url="https://api.openai.com/v1"  # Use OpenAI's base URL
)

model = OpenAIChatCompletionsModel(
    model="gpt-3.5-turbo",  # Use a valid model name
    openai_client=provider,
)

run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)

agent = Agent(
    instructions="You are a helpful assistant that can answer questions and help with tasks.",
    name="Panaversity support Agent",
    run_config=run_config  # Add the run_config parameter
)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! I am Agent, that helps human to solve there problems").send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")

    history.append({"role": "user", "content":message.content})
    result = await Runner.run(
    agent,
    input=history,
    run_config=run_config,
)
    history.append({"role":"assistant", "content":result.final_output})
    cl.user_session.set("history",history)
    await cl.Message(content=result.final_output).send()

