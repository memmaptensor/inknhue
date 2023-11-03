from PIL import Image
from torch.utils.data import Dataset


from ..utils import get_entries


#   Directory structure
#   dataset_path
#   |-  style2paints
#       |-  000000.png
#   |-  colored
#       |-  000000.png
class ConditionalDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.cond_dataset = []

        style2paints = get_entries(f"{dataset_path}/style2paints/*.png")
        colored = get_entries(f"{dataset_path}/colored/*.png")

        assert len(style2paints) == len(colored)

        for s, c in zip(style2paints, colored):
            self.cond_dataset.append({"style2paints": s, "colored": c})

    def __len__(self):
        return len(self.cond_dataset)

    def __getitem__(self, idx):
        s = Image.open(self.cond_dataset[idx]["style2paints"]).convert("RGB")
        c = Image.open(self.cond_dataset[idx]["colored"]).convert("RGB")
        g = c.convert("L").convert("RGB")

        if self.transform is not None:
            return self.transform(g, s, c)

        return g, s, c
