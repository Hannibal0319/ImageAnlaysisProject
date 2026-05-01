from dataset import MVTecDataset
root = r"C:\Users\Peter\Desktop\stuff\MIUN\Research\robustanomaly\mvtec_ad_2"
cat = "can"
ds = MVTecDataset(root, cat, split='test')
print(f"Total images: {len(ds)}")
good_count = sum(1 for l in ds.labels if l == 0)
bad_count = sum(1 for l in ds.labels if l == 1)
print(f"Good: {good_count}, Bad: {bad_count}")
if len(ds) > 0:
    print(f"First image path: {ds.image_paths[0]}")
