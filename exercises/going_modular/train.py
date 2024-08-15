import os 
import torch 
import data_setup, engine, model_builder, utils
from torchvision import transforms 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--num_epochs", help="an integer to perform number of epochs", type=int)
parser.add_argument("-b", "--batch_size", help="an integer of number of element per batch", type=int)
parser.add_argument("-hu", "--hidden_units", help="an integer of number of hidden units per layer", type=int)
parser.add_argument("-lr", "--learning_rate", help="a float for the learning rate", type=float)

args = parser.parse_args()

# setup hyperparameters 
NUM_EPOCHS = args.num_epochs if args.num_epochs else 10
BATCH_SIZE = args.batch_size # required
HIDDEN_UNITS = args.hidden_units if args.hidden_units else 10
LEARNING_RATE = args.learning_rate if args.learning_rate else 0.001

# setup directories 
train_dir = "going_modular/data/pizza_sushi_steak/train"
test_dir = "going_modular/data/pizza_sushi_steak/test"

def main():
    # setup device agnostic code 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create transforms
    data_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor()
    ])

    # create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir, 
        test_dir=test_dir, 
        transform=data_transform, 
        batch_size=BATCH_SIZE
    )

    # create model with help from model_builder.py
    model = model_builder.TinyVGG(
        input_shape=3, 
        hidden_units=HIDDEN_UNITS, 
        output_shape=len(class_names)
    ).to(device)

    # set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    # start training with help from engine.py 
    engine.train(model=model, 
                    train_dataloader=train_dataloader, 
                    test_dataloader=test_dataloader, 
                    loss_fn=loss_fn, 
                    optimizer=optimizer, 
                    epochs=NUM_EPOCHS, 
                    device=device)
            
    # save the model with help from utils.py
    utils.save_model(model=model, target_dir="models", model_name="05_going_modular_script_mode_tinyvgg_model_exc.pt")

if __name__ == '__main__':
    main()
