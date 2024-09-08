# Flux Image Generation

This repository contains a script for generating images using the Flux model
from the Hugging Face Hub, on an M1 Max Macbook pro with 32gb ram.

This is essentially a doc to help me remember how to do this again.

## Steps

1. Create a new virtual environment

```
python3 -m venv .venv
source .venv/bin/activate
```

2. Install the dependencies

```
pip install diffusers transformers torch sentencepiece accelerate protobuf
```

3. Log in to Hugging Face

```
huggingface-cli login
```

4. Run the script

```
python flux_generate.py
```
