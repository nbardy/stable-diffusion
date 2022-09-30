import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from einops import rearrange
from ldm.util import instantiate_from_config
from datasets import load_dataset


class FolderData(Dataset):
    def __init__(self, root_dir, caption_file, image_transforms, ext="jpg") -> None:
        self.root_dir = Path(root_dir + "/cache")
        with open(caption_file, "rt") as f:
            captions = json.load(f)
        self.captions = captions

        self.paths = list(self.root_dir.rglob(f"*.{ext}"))
        print("!!!!!!!!!!")
        print("!!!!!!!!!!")
        print("!!!!!!!!!!")
        print("!!!!!!!!!!")
        print("Paths")
        print("paths", self.paths)
        image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")),
            ]
        )
        image_transforms = transforms.Compose(image_transforms)
        self.tform = image_transforms

        # assert all(['full/' + str(x.name) in self.captions for x in self.paths])

    def __len__(self):
        return len(self.captions.keys())

    def __getitem__(self, index):
        chosen = list(self.captions.keys())[index]
        im = Image.open(self.root_dir / chosen)
        im = self.process_im(im)
        caption = self.captions[chosen]
        if caption is None:
            caption = "old book illustration"
        return {"jpg": im, "txt": caption}

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)


import os
import requests
import hashlib


# Downloads the image if it isn't cached
# otherwise loads from cache
#
# Returns PIL image
def fetch_image(url):
    # hash filename
    md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    # Check if cache folder exists
    if not os.path.exists("cache"):
        os.makedirs("cache")

    filename = f"cache/{md5}.jpg"
    requests.get(url, stream=True)
    if not os.path.exists(filename):
        print("Downloading", url)
        img = requests.get(url, stream=True)
        with open(filename, "wb") as f:
            f.write(img.content)
    else:
        print("Loading from cache", url)

    img = Image.open(filename)

    return img


def hf_dataset(
    name,
    image_transforms=[],
    image_url_column="image_url",
    text_column="text",
    split="train",
    image_key="image",
    caption_key="txt",
):
    """Make huggingface dataset with appropriate list of transforms applied"""
    ds = load_dataset(name, split=split)
    image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
    image_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")),
        ]
    )
    tform = transforms.Compose(image_transforms)

    assert (
        image_url_column in ds.column_names
    ), f"Didn't find column {image_url_column} in {ds.column_names}"
    assert (
        text_column in ds.column_names
    ), f"Didn't find column {text_column} in {ds.column_names}"

    def pre_process(examples):
        processed = {}
        urls = examples[image_url_column]
        images = [fetch_image(url) for url in urls]

        print(images)
        processed[image_key] = [tform(im) for im in images]
        processed[caption_key] = examples[text_column]
        return processed

    ds.set_transform(pre_process)
    return ds


class TextOnly(Dataset):
    def __init__(
        self, captions, output_size, image_key="image", caption_key="txt", n_gpus=1
    ):
        """Returns only captions with dummy images"""
        self.output_size = output_size
        self.image_key = image_key
        self.caption_key = caption_key
        if isinstance(captions, Path):
            self.captions = self._load_caption_file(captions)
        else:
            self.captions = captions

        if n_gpus > 1:
            # hack to make sure that all the captions appear on each gpu
            repeated = [n_gpus * [x] for x in self.captions]
            self.captions = []
            [self.captions.extend(x) for x in repeated]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        dummy_im = torch.zeros(3, self.output_size, self.output_size)
        dummy_im = rearrange(dummy_im * 2.0 - 1.0, "c h w -> h w c")
        return {self.image_key: dummy_im, self.caption_key: self.captions[index]}

    def _load_caption_file(self, filename):
        with open(filename, "rt") as f:
            captions = f.readlines()
        return [x.strip("\n") for x in captions]
