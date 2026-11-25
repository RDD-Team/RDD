"""Zero-DCE inference for RDD, adapted from Guo et al. (CVPR 2020).

Original model and pretrained weights: https://github.com/Li-Chongyi/Zero-DCE
Upstream implementation: CC BY-NC 4.0 (academic, non-commercial use).
"""

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn


WEIGHT_URL = (
    "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/"
    "Zero-DCE_code/snapshots/Epoch99.pth"
)
WEIGHT_SHA256 = "a4395acb874f320375d9704997cef874eaaaaa26a1777ceb29a92b70f74c3612"
EXPECTED_IMAGES = 386


class ZeroDCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.e_conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.e_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.e_conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.e_conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.e_conv5 = nn.Conv2d(64, 32, 3, padding=1)
        self.e_conv6 = nn.Conv2d(64, 32, 3, padding=1)
        self.e_conv7 = nn.Conv2d(64, 24, 3, padding=1)

    def forward(self, image):
        first = self.e_conv1(image).relu()
        second = self.e_conv2(first).relu()
        third = self.e_conv3(second).relu()
        fourth = self.e_conv4(third).relu()
        fifth = self.e_conv5(torch.cat((third, fourth), dim=1)).relu()
        sixth = self.e_conv6(torch.cat((second, fifth), dim=1)).relu()
        curves = self.e_conv7(torch.cat((first, sixth), dim=1)).tanh()
        for curve in curves.split(3, dim=1):
            image = image + curve * (image.square() - image)
        return image


def load_model(project, device):
    path = project / "weights" / "zero_dce_epoch99.pth"
    if path.is_file():
        payload = path.read_bytes()
    else:
        print("공식 Zero-DCE 가중치(Epoch99.pth)를 받는 중이에요.", flush=True)
        with urlopen(WEIGHT_URL, timeout=30) as download:
            payload = download.read(320018)
    if hashlib.sha256(payload).hexdigest() != WEIGHT_SHA256:
        raise ValueError(f"공식 Zero-DCE 가중치와 파일 내용이 달라요. {path}")
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(payload)

    model = ZeroDCE()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval()


@torch.inference_mode()
def enhance(image, model, device):
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device)
    result = model(tensor).squeeze(0).permute(1, 2, 0)
    result = (result.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    return Image.fromarray(result)


def comparison_sheet(pairs):
    tile_width, tile_height = 640, 360
    header = 38
    sheet = Image.new("RGB", (tile_width * 2, header + tile_height * len(pairs)), "#161b23")
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 12), "Original", fill="white", font_size=19)
    draw.text((tile_width + 14, 12), "Zero-DCE", fill="white", font_size=19)
    for row, (original, corrected) in enumerate(pairs):
        for column, image in enumerate((original, corrected)):
            tile = image.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
            sheet.paste(tile, (column * tile_width, header + row * tile_height))
    return sheet


def read_completed_image(image_path, label_path, expected_size, expected_label):
    if not image_path.is_file() or not label_path.is_file():
        return None
    if label_path.read_bytes() != expected_label:
        return None
    try:
        with Image.open(image_path) as image:
            if image.format == "PNG" and image.size == expected_size:
                return image.convert("RGB")
    except OSError:
        # A copy taken while the old script was writing can contain a partial PNG.
        return None
    return None


def main():
    parser = argparse.ArgumentParser(description="RDD 야간 검증 이미지 Zero-DCE 보정")
    parser.add_argument("--preview", action="store_true", help="샘플 3장의 보정 전후만 저장")
    args = parser.parse_args()

    project = Path(__file__).resolve().parent
    original_dir = project / "b2_night"
    corrected_dir = project / "b2_night_zero_dce"
    partial_dir = project / "b2_night_zero_dce_in_progress"
    if not args.preview and corrected_dir.exists():
        raise ValueError(f"이미 만든 B2 폴더가 있어요. {corrected_dir}")

    images = sorted((original_dir / "images" / "val").glob("*.png"))
    labels = original_dir / "labels" / "val"
    label_paths = {image: labels / image.with_suffix(".txt").name for image in images}
    if len(images) != EXPECTED_IMAGES or any(not path.is_file() for path in label_paths.values()):
        raise ValueError("b2_night의 평가 이미지 386장과 같은 이름의 라벨 386개를 확인해 주세요.")

    positive_images = [image for image, label in label_paths.items() if label.stat().st_size > 0]
    if len(positive_images) < 3:
        raise ValueError("보정 전후 샘플을 만들 낙하물 이미지가 부족해요.")
    sample_images = {positive_images[0], positive_images[len(positive_images) // 2], positive_images[-1]}
    selected_images = sorted(sample_images) if args.preview else images

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    print(f"Zero-DCE 보정 시작 ({device}, {len(selected_images)}장)", flush=True)
    results_dir = project / "results"
    results_dir.mkdir(exist_ok=True)
    preview_path = results_dir / "zero_dce_before_after.png"
    with TemporaryDirectory(prefix="zero_dce_", dir=project) as temporary:
        stage = Path(temporary)
        corrected_images = partial_dir / "images" / "val"
        corrected_labels = partial_dir / "labels" / "val"
        if not args.preview:
            corrected_images.mkdir(parents=True, exist_ok=True)
            corrected_labels.mkdir(parents=True, exist_ok=True)
            if {path.name for path in corrected_images.glob("*.png")} - {path.name for path in images}:
                raise ValueError("중간 저장 폴더에 평가 목록에 없는 이미지가 있어요.")
            if {path.name for path in corrected_labels.glob("*.txt")} - {path.name for path in label_paths.values()}:
                raise ValueError("중간 저장 폴더에 평가 목록에 없는 라벨이 있어요.")

        pairs = []
        reused = 0
        for done, image_path in enumerate(selected_images, 1):
            with Image.open(image_path) as original:
                original = original.convert("RGB")
            corrected = None
            if not args.preview:
                output_image = corrected_images / image_path.name
                output_label = corrected_labels / label_paths[image_path].name
                label_bytes = label_paths[image_path].read_bytes()
                corrected = read_completed_image(output_image, output_label, original.size, label_bytes)
            if corrected is None:
                if model is None:
                    model = load_model(project, device)
                corrected = enhance(original, model, device)
                if not args.preview:
                    pending_image = stage / "image.png"
                    pending_label = stage / "label.txt"
                    corrected.save(pending_image)
                    pending_label.write_bytes(label_bytes)
                    pending_image.replace(output_image)
                    pending_label.replace(output_label)
            else:
                reused += 1
            if image_path in sample_images:
                pairs.append((original, corrected))
            if done % 25 == 0 or done == len(selected_images):
                print(f"{done}/{len(selected_images)}장 완료 (이어서 사용 {reused}장)", flush=True)

        staged_preview = stage / preview_path.name
        comparison_sheet(pairs).save(staged_preview)
        if not args.preview:
            staged_yaml = stage / "data.yaml"
            staged_yaml.write_text("train: null\nval: images/val\nnames:\n  0: debris\n", encoding="utf-8")
            staged_yaml.replace(partial_dir / "data.yaml")
            partial_dir.rename(corrected_dir)
            print(f"B2 평가용 보정 이미지 저장 → {corrected_dir}")
        staged_preview.replace(preview_path)
    print(f"보정 전후 사진 저장 → {preview_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("\n보정을 중단했어요. 중간 저장된 이미지는 같은 명령으로 이어서 처리할 수 있어요.") from None
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Zero-DCE 보정 중단: {error}") from None
