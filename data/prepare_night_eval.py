import csv
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory


def find_csv(project, filename):
    matches = [path for path in (project / "metadata").rglob(filename) if path.is_file()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"metadata 폴더에 {filename}이 여러 개 있어요.")
    root_file = project / filename
    if root_file.is_file():
        return root_file
    raise ValueError(f"metadata 폴더에 {filename}을 넣어 주세요.")


def main():
    project = Path(__file__).resolve().parent
    source = project / "night_val_41776"
    destination = project / "b2_night"
    if not source.is_dir():
        raise ValueError("night_val_41776 폴더가 없어요. 야간 이미지 386장을 먼저 추출해 주세요.")
    if destination.exists():
        raise ValueError("b2_night 폴더가 이미 있어요. 기존 평가 폴더를 확인해 주세요.")

    images_csv = find_csv(project, "images.csv")
    objects_csv = find_csv(project, "objects.csv")
    print("야간 이미지와 팀 정답 좌표를 확인하는 중이에요.", flush=True)
    with images_csv.open(encoding="utf-8-sig", newline="") as stream:
        images = {
            row["image"]: row
            for row in csv.DictReader(stream)
            if (row["split"], row["time"], row["img_key"]) == ("val", "Night", "41776")
        }
    extracted = {path.name for path in source.glob("*.png") if path.is_file()}
    if len(images) != 386 or extracted != images.keys():
        raise ValueError(
            f"팀 검증 이미지 386장과 추출 폴더의 파일명이 일치해야 해요. "
            f"CSV {len(images)}장, 추출 폴더 {len(extracted)}장."
        )

    labels = {name: [] for name in images}
    with objects_csv.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["image"]
            if name not in images or row["is_debris"] != "1":
                continue
            width = int(images[name]["width"])
            height = int(images[name]["height"])
            x1, y1, x2, y2 = (float(row[field]) for field in ("x1", "y1", "x2", "y2"))
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"이미지 범위를 벗어난 정답 박스가 있어요. {name}")
            box = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                   (x2 - x1) / width, (y2 - y1) / height)
            labels[name].append("0 " + " ".join(f"{value:.8f}" for value in box))

    for name, lines in labels.items():
        if len(lines) != int(images[name]["n_debris"] or 0):
            raise ValueError(f"팀 CSV 두 파일의 정답 박스 개수가 달라요. {name}")

    with TemporaryDirectory(dir=project) as temp:
        staging = Path(temp) / "b2_night"
        image_dir = staging / "images" / "val"
        label_dir = staging / "labels" / "val"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for name, lines in labels.items():
            shutil.copy2(source / name, image_dir / name)
            label = label_dir / Path(name).with_suffix(".txt")
            label.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        staging.rename(destination)

    positives = sum(bool(lines) for lines in labels.values())
    total_boxes = sum(len(lines) for lines in labels.values())
    print(f"평가 준비 완료: 이미지 {len(images)}장, 낙하물 이미지 {positives}장, 정답 박스 {total_boxes}개")
    print(destination)


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, ValueError) as error:
        raise SystemExit(f"평가 준비 중단: {error}") from None
