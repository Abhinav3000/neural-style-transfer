import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image, ImageOps
import gradio as gr
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VGG19 with updated API
vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

# Load and pad image
def load_image(img, max_size=512):
    img = img.convert("RGB")
    img = ImageOps.pad(img, (max_size, max_size), method=Image.BICUBIC)
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(img).unsqueeze(0).to(device)

# Convert tensor to PIL
def tensor_to_pil(tensor):
    image = tensor.cpu().clone().detach().squeeze(0)
    return transforms.ToPILImage()(image.clamp(0, 1))

# Extract features
def get_features(image, model, style_layers, content_layer):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in style_layers + [content_layer]:
            features[name] = x
    return features

# Gram matrix for style
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    return torch.mm(features, features.t()) / (c * h * w)

# Style Transfer Core
def style_transfer(content_img, style_img, steps, style_weight, size):
    style_layers = ['0', '5', '10', '19', '28']
    content_layer = '21'

    content = load_image(content_img, size)
    style = load_image(style_img, size)

    content_features = get_features(content, vgg, style_layers, content_layer)
    style_features = get_features(style, vgg, style_layers, content_layer)

    style_grams = [gram_matrix(style_features[layer]) for layer in style_layers]
    content_target = content_features[content_layer]

    generated = content.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([generated])
    run = [0]

    def closure():
        optimizer.zero_grad()
        gen_features = get_features(generated, vgg, style_layers, content_layer)
        style_outputs = [gen_features[layer] for layer in style_layers]
        content_output = gen_features[content_layer]
        content_loss = F.mse_loss(content_output, content_target)
        style_loss = sum(F.mse_loss(gram_matrix(f), t) for f, t in zip(style_outputs, style_grams))
        total_loss = content_loss + style_weight * style_loss
        total_loss.backward()
        run[0] += 1
        return total_loss

    while run[0] <= steps:
        optimizer.step(closure)

    return tensor_to_pil(generated)

# Preloaded styles
def get_style_img(name):
    return Image.open(f"styles/{name}")

style_dict = {
    "Starry Night (Van Gogh)": "starry_night.jpg",
    "Water Lilies (Monet)": "water_lilies.jpg",
    "Madhubani (Indian Folk Art)": "madhubani.jpg",
    "Upload Your Own": None
}

# Unified function for Gradio
def style_transfer_ui(content_img, style_choice, uploaded_style_img, steps, style_weight, size):
    selected = style_choice if isinstance(style_choice, str) else style_choice[0]
    if style_dict[selected] is None:
        style_img = uploaded_style_img
    else:
        style_img = get_style_img(style_dict[selected])
    return style_transfer(content_img, style_img, steps, style_weight, size)

# Gradio App
app = gr.Interface(
    fn=style_transfer_ui,
    inputs=[
        gr.Image(type="pil", label="📷 Content Image"),
        gr.Dropdown(choices=list(style_dict.keys()), label="🎨 Choose a Style", multiselect=False),
        gr.Image(type="pil", label="🖼 Upload Style (if using 'Upload Your Own')"),
        gr.Slider(50, 500, step=10, value=300, label="💎 Steps"),
        gr.Slider(100, 10000, step=100, value=5000, label="🎯 Style Weight"),
        gr.Slider(256, 768, step=64, value=512, label="📐 Image Size")
    ],
    outputs=gr.Image(type="pil", label="🖌 Stylized Output"),
    title="🎨 Neural Style Transfer – Smart Resume Version",
    description="Upload a content image and choose a preloaded or custom style. Tweak steps, style weight, and output size!"
)

if __name__ == "__main__":
    app.launch()
