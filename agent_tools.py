# Agent setup
from langchain.agents import tool
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  AIMessagePromptTemplate,
  HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
# Coingecko Linkup
from pycoingecko import CoinGeckoAPI
import time
import cache
import image_generator


# Agent Tools
# Stability AI Image Generation Functions
@tool("generate_image", return_direct=True)
def generate_image(input):
  """Useful for when an image is specifically asked to be generated or created from Stability AI's platform. The input is the image that the user is describing. """
  result = {}
  result['output'] = ""
  result['image_generated'] = ""
  system_prompt = "Stable Diffusion is an AI art generation model similar to DALLE-2. \
Below is a list of prompts that can be used to generate images with Stable Diffusion:\
- portait of a homer simpson archer shooting arrow at forest monster, front game card, drark, marvel comics, dark, intricate, highly detailed, smooth, artstation, digital illustration by ruan jia and mandy jurgens and artgerm and wayne barlowe and greg rutkowski and zdislav beksinski \
- pirate, concept art, deep focus, fantasy, intricate, highly detailed, digital painting, artstation, matte, sharp focus, illustration, art by magali villeneuve, chippy, ryan yee, rk post, clint cearley, daniel ljunggren, zoltan boros, gabor szikszai, howard lyon, steve argyle, winona nelson \
- ghost inside a hunted room, art by lois van baarle and loish and ross tran and rossdraws and sam yang and samdoesarts and artgerm, digital art, highly detailed, intricate, sharp focus, Trending on Artstation HQ, deviantart, unreal engine 5, 4K UHD image \
- red dead redemption 2, cinematic view, epic sky, detailed, concept art, low angle, high detail, warm lighting, volumetric, godrays, vivid, beautiful, trending on artstation, by jordan grimmer, huge scene, grass, art greg rutkowski \
- a fantasy style portrait painting of rachel lane / alison brie hybrid in the style of francois boucher oil painting unreal 5 daz. rpg portrait, extremely detailed artgerm greg rutkowski alphonse mucha greg hildebrandt tim hildebrandt\
- athena, greek goddess, claudia black, art by artgerm and greg rutkowski and magali villeneuve, bronze greek armor, owl crown, d & d, fantasy, intricate, portrait, highly detailed, headshot, digital painting, trending on artstation, concept art, sharp focus, illustration \
- closeup portrait shot of a large strong female biomechanic woman in a scenic scifi environment, intricate, elegant, highly detailed, centered, digital painting, artstation, concept art, smooth, sharp focus, warframe, illustration, thomas kinkade, tomasz alen kopera, peter mohrbacher, donato giancola, leyendecker, boris vallejo \
- ultra realistic illustration of steve urkle as the hulk, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha \
I want you to write me a single detailed prompt exactly about the idea written after IDEA. Follow the structure of the example prompts. This means a very short description of the scene, followed by modifiers divided by commas to alter the mood, style, lighting, and more. \
IDEA: {input} \n Prompt: "

  prompt = PromptTemplate(template=system_prompt, input_variables=["input"])
  llm_chain = LLMChain(prompt=prompt,
                       llm=ChatOpenAI(temperature=0.7),
                       verbose=True)
  response = llm_chain.predict(input=input)
  print(response)
  new_prompt = response.split("Prompt:")[-1]
  result = image_generator.get_image_from_stability(new_prompt)
  return result


@tool("code_snippets", return_direct=True)
def code_snippets(query: str) -> str:
  """Useful for generating code snippets. The input should be the query in natural language."""
  template = """Q:{question}"""
  prompt = PromptTemplate(template=template, input_variables=["question"])
  llm_chain = LLMChain(prompt=prompt,
                       llm=ChatOpenAI(temperature=0.7),
                       verbose=True)

  response = llm_chain.predict(question=query)
  return response


@tool("web3_token_price", return_direct=False)
def web3_token_price(query: str) -> str:
  """Useful for when you want to ask questions about price, market cap and market volume of cryptocurrencies. The input should be the name of the coin or token."""
  st = time.time()
  token_price_header = "Game, Token, Token Price, Token Market Cap, Token Market Volume\n"
  game_list = query.split(",")
  print(game_list)
  token_stats = ""
  for g in game_list:
    token_stats += get_token_price(g)
  print(token_stats)
  print("End of Web3 Token Price Search: " + str(time.time() - st) +
        " seconds")
  return token_price_header + token_stats


def get_token_price(token_name: str) -> str:
  full_stat = ""
  try:
    cached = cache.get_cache(token_name)
    if cached:
      return cached
    token_symbol = token_name
    cg = CoinGeckoAPI()

    # Check for AVG. Because AVG pulls up SAVG instead, specify the AvocadoDAO coin directly
    if token_symbol.lower() == "avg":
      token_symbol = "avocado dao"
    token_data = cg.search(token_symbol.lower())
    if token_data:
      # print("Token Data: " + str(token_data))
      if len(token_data['coins']) > 0:
        token_api_symbol = token_data['coins'][0]['api_symbol']
        token_stats = cg.get_price(ids=token_data['coins'][0]['api_symbol'],
                                   vs_currencies='usd',
                                   include_market_cap=True,
                                   include_24hr_vol=True)

        if token_stats:
          token_mcap = "{:,.0f}".format(
            token_stats[token_api_symbol]['usd_market_cap'])
          token_volume = "{:,.0f}".format(
            token_stats[token_api_symbol]['usd_24h_vol'])
          token_price = "{:,.2f}".format(token_stats[token_api_symbol]['usd'])
          full_stat = f"{token_symbol}, USD{token_price}, USD{token_mcap}, USD{token_volume}"
          cache.set_cache(token_name, full_stat)
          return f"{full_stat}\n"
  except Exception as e:
    print(f"web3_token_price:{e}")
    return f"No token prices available for {token_name}.\n"

  return f"No token prices available for {token_name}.\n"


@tool("google_search", return_direct=False)
def google_search(query: str) -> str:
  """Useful for when you need to answer questions about current events. The input is the search query."""
  search = GoogleSerperAPIWrapper()
  response = search.run(query)
  return response
