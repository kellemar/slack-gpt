import openai
import os
import time
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient

import traceback

# Langchain setup
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  AIMessagePromptTemplate,
  HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Agents
from langchain import LLMChain, PromptTemplate
from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.agents import initialize_agent
import agent_tools

#JSON
import json

# Configuration
INTENTS = """market cap", "volume", "token", "price", "availability", "create image", "download"""

DEFAULT_PROMPT = (
  "You are a friendly assistant called 'AvodudeAI', for a company that can answer general questions. Your goal is to help the people in "
  "the company with any questions they might have.  If you aren't sure about something, you should say that you don't know."
)

ADDITIONS = """Only answer in the following JSON structure {"answer": "..","subject_matter": [".."],"intents": [".."]} For the "intents" field, choose from the following values that best describes the intents: """ + INTENTS + "\n\nContext:\n"

# The OpenAI model to use. Can be gpt-3.5-turbo or gpt-4.
MODEL = os.environ.get("MODEL", "gpt-3.5-turbo")
# The max length of a message to OpenAI.
MAX_TOKENS = 8000 if MODEL == "gpt-4" else 4096
# The max length of a response from OpenAI.
MAX_RESPONSE_TOKENS = 1000
# Starts with "sk-", used for connecting to OpenAI.
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
# Starts with "xapp-", used for connecting to Slack.
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
# Starts with "xoxb-", used for connecting to Slack.
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
# Tokens are ~4 characters but this script doens't account for that yet.
TOKEN_MULTIPLIER = 4

# Initialize the Slack Bolt App and Slack Web Client
app = App()
slack_client = WebClient(token=SLACK_BOT_TOKEN)

# Set up the default prompt and OpenAI API
prompt = os.environ.get("PROMPT", DEFAULT_PROMPT) + ADDITIONS
openai.api_key = OPENAI_API_KEY
llm_chat = ChatOpenAI(temperature=0.7, max_tokens=MAX_RESPONSE_TOKENS)

tools = load_tools(["wikipedia"], llm=llm_chat)

# This is done to make sure that other tools are loaded, and I don't really need Wikipedia since google search does the same job.
tools.pop(0)
tools.append(agent_tools.google_search)
tools.append(agent_tools.web3_token_price)
tools.append(agent_tools.generate_image)
tools.append(agent_tools.code_snippets)

def generate_completion_langchain(prompt, messages, query):
  google_agent = initialize_agent(
    tools,
    llm_chat,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    memory=messages)
  completion = google_agent.run(input=DEFAULT_PROMPT + query)
  return completion


# Supposed used for refining answers due to errors. But not really used.
def refine_answer(query: str) -> str:
  output = {}
  template = """Summarize this text to make sure it's properly readable by the average human. \nQ:{question}"""
  prompt = PromptTemplate(template=template, input_variables=["question"])
  llm_chain = LLMChain(prompt=prompt,
                       llm=ChatOpenAI(temperature=0.7),
                       verbose=True)

  response = llm_chain.predict(question=query)
  return response


def get_message_history(channel_id, user_id, event_ts, limit, thread=False):
  """Fetch conversation or thread history and build a list of messages."""
  history = []

  # This starts chat history for each individual user.
  memory = ConversationBufferMemory(memory_key="chat_history",
                                    return_messages=True)

  # Fetch the message history
  if thread:
    result = slack_client.conversations_replies(channel=channel_id,
                                                ts=event_ts,
                                                limit=limit,
                                                latest=int(time.time()))
  else:
    result = slack_client.conversations_history(channel=channel_id,
                                                limit=limit)

  token_count = 0

  for message in result["messages"]:

    if message.get("user") == user_id:
      role = "user"
    # You change this to how it should check for image uploads from the bot.
    elif ("files" in message and message["files"][0]["title"]) == "AvodudeAI Image Generator" or message.get("subtype") == "bot_message" or message.get("bot_id"):
      role = "assistant"
    else:
      continue

    # Ignore typing responses from the bot
    if message["text"] == "Typing a response...":
      continue
    token_count += len(message["text"])
    if token_count > (MAX_TOKENS - MAX_RESPONSE_TOKENS):
      break
    else:
      try:
        
        if role == "user":
          memory.chat_memory.add_user_message(message["text"])
        else:
          memory.chat_memory.add_ai_message(message["text"])
        history.append({"role": role, "content": message["text"]})
      except Exception as e:
        traceback.print_exc()
        

  # DMs are in reverse order while threads are not.
  if not thread:
    history.reverse()
  return memory, history


def handle_message(event, thread=False):
  """Handle a direct message or mention."""
  channel_id = event["channel"]
  user_id = event["user"]
  event_ts = event["ts"]

  # Set up the payload for the "Typing a response..." message
  payload = {"channel": channel_id, "text": "Typing a response..."}

  if thread:
    # Use the thread_ts as the event_ts when in a thread
    event_ts = event.get("thread_ts", event["ts"])
    payload["thread_ts"] = event_ts

  # Get message history
  chat_history, history = get_message_history(channel_id,
                                              user_id,
                                              event_ts,
                                              limit=8,
                                              thread=thread)
  actual_query = history[-1]['content']
  # Send "Typing a response..." message
  typing_message = slack_client.chat_postMessage(**payload)

  # Generate the completion
  try:
    completion_message = generate_completion_langchain(prompt, chat_history,
                                                       actual_query)
  except Exception as e:
    traceback.print_exc()
    completion_message = (
      "Something happened when trying to answer your query. Please try again.")

  # Replace "Typing a response..." with the actual response
  # Normal text response
  if 'File:' in completion_message:
    split_message = completion_message.split("File:")
    slack_client.files_upload_v2(channel=channel_id,
                                 title="AvodudeAI Image Generator",
                                 file=split_message[1],
                                 initial_comment=split_message[0])

  else:
    slack_client.chat_update(channel=channel_id,
                             ts=typing_message["ts"],
                             text=completion_message)


@app.event("app_mention")
def mention_handler(body, say):
  """Handle app mention events."""
  event = body["event"]
  handle_message(event, thread=True)


@app.event("message")
def direct_message_handler(body, say):
  """Handle direct message events."""
  event = body["event"]
  if event.get("subtype") == "bot_message" or event.get("bot_id"):
    return
  handle_message(event)


if __name__ == "__main__":
  handler = SocketModeHandler(app, SLACK_APP_TOKEN)
  handler.start()
