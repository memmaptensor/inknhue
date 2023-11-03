import torch
from PIL import Image
from torchvision.transforms import ToTensor

from .models import Colorizer
from .utils import resize_pad


class Shader:
    def __init__(
        self,
        device,
        generator_path="./shader.ckpt",
    ):
        self.colorizer = Colorizer().to(device)
        self.colorizer.generator.load_state_dict(
            torch.load(generator_path, map_location=device)
        )
        self.colorizer = self.colorizer.eval()
        self.device = device

    def shade(self, image, size=576, transform=ToTensor()):
        image, current_pad = resize_pad(image, size)
        current_image = transform(image).unsqueeze(0).to(self.device)
        current_hint = (
            torch.zeros(1, 4, current_image.shape[2], current_image.shape[3])
            .float()
            .to(self.device)
        )

        with torch.no_grad():
            fake_color, _ = self.colorizer(torch.cat([current_image, current_hint], 1))
            fake_color = fake_color.detach()

        result = fake_color[0].cpu().permute(1, 2, 0) * 0.5 + 0.5
        result = (result * 255.0).clamp(0, 255)

        if current_pad[0] != 0:
            result = result[: -current_pad[0]]
        if current_pad[1] != 0:
            result = result[:, : -current_pad[1]]

        result = Image.fromarray(result.to(dtype=torch.uint8).numpy())
        result = result.convert("L").convert("RGB")

        return result
