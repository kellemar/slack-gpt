import os
# Set up in-memory cache
import redis
import langchain
from langchain.cache import InMemoryCache
#langchain.llm_cache = InMemoryCache()

# Set up Redis Cache
# Initialize Redis
r = redis.Redis(host=os.environ["REDIS_URL"],
                port='40130',
                password=os.environ.get("REDIS_PASSWORD"),
                ssl=True,
                decode_responses=True)

# Set caching functions
def set_cache(key, value, ttl=1800, redis_cache=r):
  cached_key = key.lower().replace(" ", "_")
  redis_cache.set("cached_" + cached_key, str(value), ex=ttl)
  return "OK"


def get_cache(key, redis_cache=r):
  cached_key = key.lower().replace(" ", "_")
  cached = redis_cache.get("cached_" + cached_key)
  if cached:
    return cached
  else:
    return 0