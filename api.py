from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from starlette.status import HTTP_302_FOUND
from urllib.parse import urlparse
from pprint import pprint as pp
import json
import sys

from birdseye import eye
import builtins
builtins.eye = eye

origins = [
    "http://localhost",
    "http://localhost:1234",
    "http://localhost:8080",
    "http://docs.gpt4.org",
    "http://t70.0.gpt4.org",
]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"Hello": "FastAPI"}

import re

@eye
def resolve(fqn, fname=''):
    fqn = fqn.replace('._src.', '.')
    fqn = fqn.replace('jax.lax.lax.', 'jax.lax.')
    fqn = fqn.replace('jax.numpy.lax_numpy.', 'jax.numpy.')
    fqn = fqn.replace('jax.nn.functions.', 'jax.nn.')
    print(fqn)
    module, name = fqn.rsplit('.', 1)
    root = module.split('.', 1)[0]
    print(module, name)
    #url = f'http://some.other.api/{fqn}'
    if root == 'numpy':
      name = ({
        'sometrue': 'any',
        'alltrue': 'all',
        'cumproduct': 'cumprod',
        'product': 'prod',
        'round_': 'around',
        }).get(name, name)
      return f'https://numpy.org/doc/stable/reference/generated/{module}.{name}.html#{module}.{name}'
    if root == 'ray':
      return f'https://docs.ray.io/en/master/package-ref.html#{fqn}'
    if root == 'haiku':
      return f'https://dm-haiku.readthedocs.io/en/latest/api.html#{fqn}'
    if root == 'optax':
      return f'https://optax.readthedocs.io/en/latest/api.html#{root}.{name}'
    if root == 'jax':
      if module.count('.') <= 0:
        if name.startswith('tree_'):
          return resolve(f'jax.tree_util.{name}', fname=name)
        else:
          return f'https://jax.readthedocs.io/en/latest/jax.html#jax.{name}'
      else:
        lib = module.split('.')[1]
        #if len(re.findall(r'jax.numpy\b', fqn)) > 0:
        if lib == 'experimental':
          return f'https://jax.readthedocs.io/en/latest/{module}.html#{fqn}'
        else:
          return f'https://jax.readthedocs.io/en/latest/_autosummary/jax.{lib}.{name}.html#jax.{lib}.{name}'
      # if len(re.findall(r'jax.numpy\b', fqn)) > 0:
      #   return f'https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.{name}.html#jax.numpy.{name}'
  

@app.get("/api")
async def get_docs(request: Request):
    params = dict(request.query_params.items())
    fqn = params['id']
    fn = params.get('function.name', '')
    url = resolve(fqn, fname=fn)
    if url is not None:
      pp({"redirect": url, "params": params})
      #headers = {'Authorization': "some_long_key"}
      headers = {}
      response = RedirectResponse(url=url, headers=headers, status_code=HTTP_302_FOUND)
      return response
    return {"error": "not found", "params": params}

