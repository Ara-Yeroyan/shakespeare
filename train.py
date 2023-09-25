import argparse
from loguru import logger
from transformers import Trainer, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling

from utils import get_model_tokenizer, launch_tensorboard


def train_model(train_file, test_file, output_dir, model_name, num_train_epochs, per_device_train_batch_size,
                save_steps,port):

    model, tokenizer = get_model_tokenizer(model_name)
    eval_dataset = TextDataset(tokenizer=tokenizer, file_path=test_file, block_size=1024)
    train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_file, block_size=1024)
    data_col = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_args = TrainingArguments(
        save_steps=save_steps,
        save_total_limit=2,
        num_train_epochs=num_train_epochs,
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy='epoch',
        per_device_train_batch_size=per_device_train_batch_size
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_col,
        eval_dataset=eval_dataset,
        train_dataset=train_dataset
    )

    launch_tensorboard(log_dir='experiments', port=port)

    logger.debug(model.device)
    out = trainer.train()
    if out:
        logger.debug(out)

    trainer.save_model()
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT-2 model")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test data file")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Pretrained model name or path")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--port", type=int, default=6666, help="Tensorboard port")
    parser.add_argument("--per_device_train_batch_size", type=int, default=48, help="Batch size per device")
    parser.add_argument("--save_steps", type=int, default=10_000, help="Save model checkpoints every N steps")

    args = parser.parse_args()

    train_model(
        train_file=args.train_file,
        test_file=args.test_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        save_steps=args.save_steps,
        port=args.port
    )
