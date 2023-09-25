import os


from flask import Flask, request, jsonify

from train import train_model
from utils import generate_text, get_model_tokenizer


trained = False
app = Flask(__name__)


inference_model_name = os.environ.get("CHECKPOINT_DIR", "experiments")
if os.path.exists(inference_model_name):
    inference_model, inference_tokenizer = get_model_tokenizer(inference_model_name)  # load and store in memory
else:
    inference_model, inference_tokenizer = None, None
# except OSError:
#     print('No local checkpoint found, please train a model!')
#     inference_model, inference_tokenizer = get_model_tokenizer("gpt2")


@app.route('/train', methods=['POST'])
def train():
    try:
        args = request.json
        test_file = args['test_file']
        train_file = args['train_file']
        output_dir = args['output_dir']

        port = args.get("port", 6666)
        model_name=args.get("model_name", "gpt2")
        num_train_epochs = args.get("num_train_epochs", 5)
        per_device_train_batch_size = args.get("per_device_train_batch_size", 8)
        save_steps = args.get("save_steps", 10_000)

        train_model(
            train_file=train_file,
            test_file=test_file,
            output_dir=output_dir,
            model_name=model_name,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            save_steps=save_steps,
            port=port
        )

        return jsonify({'Status': 'Successfully Trained'})

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/generate_response', methods=['POST'])
def generate_response():
    try:
        input_text = request.json['input_text']
        max_length = request.json.get('max_length', 100)
        inference_params = request.json.get('inference_params', {'top_p': 0.9})

        global inference_model
        global inference_tokenizer

        if not inference_model:
            inference_model_name = os.environ.get("CHECKPOINT_DIR", "experiments")
            inference_model, inference_tokenizer = get_model_tokenizer(inference_model_name)

        generated_text = generate_text(inference_model,
                                       inference_tokenizer,
                                       input_text=input_text,
                                       max_length=max_length,
                                       **inference_params)
        print(generated_text)
        return jsonify({'generated_text': generated_text})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    port = os.environ.get('PORT', '5000')
    host = os.environ.get('HOST', '0.0.0.0')
    app.run(host=host, port=int(port), debug=False)