import argparse
import torch 
import matplotlib.pyplot as plt 
import requests
from PIL import Image
from torchvision import transforms
import data_setup, model_builder

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", help="string of url to the image", type=str)
args = parser.parse_args()

URL = args.image # required

tf = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor()
])

# setup directories 
train_dir = "going_modular/data/pizza_sushi_steak/train"
test_dir = "going_modular/data/pizza_sushi_steak/test"

_, _, class_names = data_setup.create_dataloaders(train_dir=train_dir, test_dir=test_dir, transform=tf, batch_size=32)

# load saved model 
loaded_model = model_builder.TinyVGG(input_shape=3, hidden_units=128, output_shape=len(class_names))
loaded_model.load_state_dict(torch.load("models/05_going_modular_script_mode_tinyvgg_model_exc.pt", weights_only=True))

def pred_and_plot(model: torch.nn.Module, 
                    image_path: str,
                    transform: transforms.Compose,
                    class_names: list[str] = None):
        # load image
        img = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
        # setup transformed image 
        transformed_img = transform(img)
        # forward pass 
        logits = model(transformed_img.unsqueeze(dim=0))
        pred = torch.softmax(logits, dim=-1).argmax(dim=-1)
        # plot the image along with the label 
        plt.imshow(transformed_img.permute(1, 2, 0))
        title = f"{class_names[pred]} | {torch.softmax(logits, dim=-1).squeeze()[pred].item():.3f}"
        plt.title(title)
        print(title)

pred_and_plot(model=loaded_model, image_path=URL, 
                transform=tf, class_names=class_names)
