import os

dataset_path = "dataset"
EXT = (".jpg", ".jpeg", ".png")

total = 0
classes = 0
for name in sorted(os.listdir(dataset_path)):
    path = os.path.join(dataset_path, name)
    if os.path.isdir(path):
        files = os.listdir(path)
        images = sum(f.lower().endswith(EXT) for f in files)
        others = len(files) - images
        note = ""
        if images < 200:
            note += "  <-- few images"
        if others:
            note += f"  <-- {others} non-image items (subfolder?)"
        print(f"{name:55s} {images:6d}{note}")
        total += images
        classes += 1

print(f"\nClasses: {classes} | Images: {total}")