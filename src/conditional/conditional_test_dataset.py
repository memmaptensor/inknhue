from PIL import Image
from torch.utils.data import Dataset


from ..utils import get_entries


#   Directory structure
#   dataset_path
#   |-  grayscale
#       |-  000000.png
#   |-  style2paints
#       |-  000000.png
class ConditionalTestDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.cond_dataset = []

        grayscale = get_entries(f"{dataset_path}/grayscale/*.png")
        style2paints = get_entries(f"{dataset_path}/style2paints/*.png")

        assert len(grayscale) == len(style2paints)

        for g, s in zip(grayscale, style2paints):
            self.cond_dataset.append({"grayscale": g, "style2paints": s})

    def __len__(self):
        return len(self.cond_dataset)

    def __getitem__(self, idx):
        g = Image.open(self.cond_dataset[idx]["grayscale"]).convert("RGB")
        s = Image.open(self.cond_dataset[idx]["style2paints"]).convert("RGB")

        if self.transform is not None:
            return self.transform(g, s)

        return g, s
