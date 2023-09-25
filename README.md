# shakespeare
Custom LLM to speake in a shakespeare style

## Usage

### Build the Environment

There are two different requirements (the one for docker does not contain torch, 
as it is included in the base image)
```bash
pip install -r requirements.txt
```

### Extract and Preprocess (Option1 from DataProcessing notebook)

```python
python extract_and_process.py
```

Run the Flask App
```python
python main.py
```

### Train

```python
python train.py --train_file data/enhanced/train.txt --test_file data/enhanced/test.txt --output_dir "experiments" --model_name gpt2 --num_train_epochs 5 --per_device_train_batch_size 16 --save_steps 10000
```
For the additional info, please, refer to the **train.py** script with provided descriptions or use help

***or***

```bash
curl -X POST -H "Content-Type: application/json" -d "{\"test_file\": \"data/enhanced/test.txt\", \"train_file\": \"data/enhanced/test.txt\", \"output_dir\": \"experiments\", \"port\": 6666, \"model_name\": \"gpt2\", \"num_train_epochs\": 5, \"per_device_train_batch_size\": 8, \"save_steps\": 10000}" http://localhost:5000/train
```

### Inference

If you do not have trained model (e.g. when firstly interacting with docker container), you need to train. 
Then during the first call to the inference endpoint, the model will be loaded and stored in memory (globally) 
Hence, after that cold start, or requests will be veyr fast!

```bash
curl -X POST -H "Content-Type: application/json" -d "{\"input_text\": \"To be or not to be, that is the question:\"}" http://localhost:5000/generate_text
```

## Docker

Build the Image

1. Downloads the data
2. Downloads the gpt2 related models/tokenizers

```bash
docker build -t my-gpt2-inference-image .
```

Run the container that supports train and inference methods

```bash
docker run --name test_gpt -p 6666:6666 --gpus all gpt2_inference_image
```

Run the same commands (curl) for train and inference 

### Note

You can also use volumes to store training results
Just create a named volume

```bash
docker volume create checkpoint
```

And attach  to the container

```bash
docker run --name test_gpt  -v checkpoint:/app/experiments -p 6666:6666 -p 5000:5000 --gpus all gpt2_inference_image
```