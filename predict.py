import struct
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from main import Net  # 确保main.py中有Net模型的定义


def read_idx_images(filepath):
    with open(filepath, 'rb') as file:
        magic, num, rows, cols = struct.unpack('>IIII', file.read(16))
        images = np.fromfile(file, dtype=np.uint8).reshape(num, 28, 28)
    return images


def read_idx_labels(filepath):
    with open(filepath, 'rb') as file:
        magic, num = struct.unpack('>II', file.read(8))
        labels = np.fromfile(file, dtype=np.uint8)
    return labels


def get_test_data(images_path, labels_path):
    images = read_idx_images(images_path)
    labels = read_idx_labels(labels_path)

    tensor_images = torch.tensor(images, dtype=torch.float32) / 255.0
    tensor_images = tensor_images.unsqueeze(1)  # 添加通道维度

    return [(img, label) for img, label in zip(tensor_images, labels)]


def load_model(model_path, device):
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = ImageOps.invert(image)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)  # 使用Image.Resampling.LANCZOS替换Image.ANTIALIAS

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image_tensor = transform(image)

    return image_tensor


def predict_external_image(model, device, image_path):
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        prediction = output.argmax(dim=1, keepdim=True).item()

    return image_tensor, prediction


def visualize(image, true_label, predicted_label, is_external):
    if is_external:
        plt.imshow(image, cmap='gray')  # 外部图片直接显示
    else:
        plt.imshow(image.squeeze(), cmap='gray')  # 数据集图片需要squeeze()

    title = f'Predicted: {predicted_label}'
    if not is_external:
        title += f', True Label: {true_label}'
        if true_label == predicted_label:
            title += ' (Correct)'
        else:
            title += ' (Incorrect)'
    plt.title(title)
    plt.show()


def main(model=None, device=None):
    if model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model('mnist_cnn.pt', device)

    images_path = '../data/MNIST/raw/train-images-idx3-ubyte'
    labels_path = '../data/MNIST/raw/train-labels-idx1-ubyte'
    test_dataset = get_test_data(images_path, labels_path)

    while True:
        choice = input("Predict 'external' image or 'dataset' image? [external/dataset/exit]: ")
        if choice.lower() == 'exit':
            break
        elif choice.lower() == 'external':
            image_path = input("Enter the path of the external image: ")
            image_tensor, prediction = predict_external_image(model, device, image_path)

            # 将Tensor转换回PIL图像以显示
            image_pil = transforms.ToPILImage()(image_tensor.squeeze())
            visualize(image_pil, None, prediction, True)  # 显示外部图片
        elif choice.lower() == 'dataset':
            index = int(input("Enter test image index (0-9999): "))
            image, true_label = test_dataset[index]
            image_tensor = image.unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(image_tensor)
                prediction = output.argmax(dim=1, keepdim=True).item()
            visualize(image.squeeze(), true_label, prediction, False)  # 显示数据集图片
        else:
            print("Invalid choice. Please enter 'external', 'dataset', or 'exit'.")

        plt.close()


if __name__ == '__main__':
    main()
