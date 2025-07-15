import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VGG19 model
vgg = models.vgg19(pretrained=True).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

# Image loader
def image_loader(image, max_size=400):
    image = image.convert('RGB')
    size = max(image.size) if max(image.size) < max_size else max_size
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float)

# Gram Matrix
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(c * h * w)

# Feature extraction
def get_features(image, model, style_layers, content_layer):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in style_layers + [content_layer]:
            features[name] = x
    return features

# Style transfer logic
def style_transfer(content_img, style_img, steps=300):
    style_layers = ['0', '5', '10', '19', '28']
    content_layer = '21'

    content_img = image_loader(content_img)
    style_img = image_loader(style_img, max_size=content_img.shape[-1])

    content_features = get_features(content_img, vgg, style_layers, content_layer)
    style_features = get_features(style_img, vgg, style_layers, content_layer)
    style_grams = [gram_matrix(style_features[layer]) for layer in style_layers]
    content_target = content_features[content_layer]

    generated = content_img.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([generated])
    run = [0]

    def closure():
        optimizer.zero_grad()
        gen_features = get_features(generated, vgg, style_layers, content_layer)
        style_outputs = [gen_features[layer] for layer in style_layers]
        content_output = gen_features[content_layer]
        content_loss = F.mse_loss(content_output, content_target)
        style_loss = sum(F.mse_loss(gram_matrix(f), t) for f, t in zip(style_outputs, style_grams))
        loss = content_loss + 1000 * style_loss
        loss.backward()
        run[0] += 1
        return loss

    while run[0] <= steps:
        optimizer.step(closure)

    final_img = generated.cpu().clone().squeeze(0)
    final_img = transforms.ToPILImage()(final_img)
    return final_img

# Gradio Interface
app = gr.Interface(
    fn=style_transfer,
    inputs=[
        gr.Image(type="pil", label="Content Image"),
        gr.Image(type="pil", label="Style Image")
    ],
    outputs=gr.Image(type="pil", label="Stylized Output"),
    title="🎨 Neural Style Transfer",
    description="Upload a content image and a style image to blend them with deep learning magic."
)

if __name__ == "__main__":
    app.launch()
