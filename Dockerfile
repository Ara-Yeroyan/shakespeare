# Use a base image with Python and PyTorch pre-installed
FROM pytorch/pytorch:latest

WORKDIR /app

COPY requirements_docker.txt requirements_docker.txt
RUN pip install -r requirements_docker.txt

# download the data once during image creation, reuse in containers
COPY extract_and_process.py extract_and_process.py
RUN python extract_and_process.py

# download the models once during image creation, reuse in containers
RUN python -c "from transformers import GPT2LMHeadModel, GPT2Tokenizer;  \
    model = GPT2LMHeadModel.from_pretrained('gpt2'); tokenizer = GPT2Tokenizer.from_pretrained('gpt2')"

# most commonly changed files - requirements will be cached
COPY utils.py utils.py
COPY train.py train.py
COPY main.py main.py

CMD ["python", "main.py"]