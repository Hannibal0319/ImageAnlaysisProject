import os
import numpy as np
from PIL import Image
from pathlib import Path
import json

def get_dataset_stats(root_path):
    root = Path(root_path)
    stats = {
        "categories": {},
        "total_images": 0,
        "total_train": 0,
        "total_test": 0,
        "resolutions": set(),
        "mean_pixel_value": [],
        "std_pixel_value": []
    }
    
    categories = [d for d in root.iterdir() if d.is_dir()]
    
    # For robustness tag analysis
    stats["tags"] = {}
    
    for cat in categories:
        print(f"  Processing category: {cat.name}", flush=True)
        cat_stats = {
            "train_good": 0,
            "test": {},
            "test_total": 0,
            "resolutions": set(),
            "contrast": [],
            "mask_percentage": []
        }
        
        # Train stats
        train_dir = cat / "train" / "good"
        if train_dir.exists():
            train_files = list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpg"))
            cat_stats["train_good"] = len(train_files)
            stats["total_train"] += len(train_files)
            
            for img_path in train_files[:10]:
                try:
                    with Image.open(img_path) as img:
                        cat_stats["resolutions"].add(img.size)
                        stats["resolutions"].add(img.size)
                        img_arr = np.array(img).astype(float) / 255.0
                        
                        # Contrast (RMS)
                        cat_stats["contrast"].append(np.std(img_arr))
                        
                        if len(img_arr.shape) == 2:
                            mean = np.mean(img_arr)
                            std = np.std(img_arr)
                            stats["mean_pixel_value"].append([mean, mean, mean])
                            stats["std_pixel_value"].append([std, std, std])
                        else:
                            stats["mean_pixel_value"].append(np.mean(img_arr, axis=(0, 1)).tolist())
                            stats["std_pixel_value"].append(np.std(img_arr, axis=(0, 1)).tolist())
                except Exception as e:
                    print(f"Error processing {img_path}: {e}", flush=True)
        
        # Test stats and Tags
        test_folders = ["test", "test_public", "validation"]
        for tf in test_folders:
            tf_dir = cat / tf
            if tf_dir.exists():
                for anomaly_type_dir in tf_dir.iterdir():
                    if anomaly_type_dir.is_dir():
                        test_files = list(anomaly_type_dir.glob("*.png")) + list(anomaly_type_dir.glob("*.jpg"))
                        cat_stats["test"][f"{tf}/{anomaly_type_dir.name}"] = len(test_files)
                        cat_stats["test_total"] += len(test_files)
                        stats["total_test"] += len(test_files)
                        
                        for f in test_files:
                            # Extract tags like 'overexposed', 'shift'
                            parts = f.stem.split("_")
                            for p in parts:
                                if p.isalpha() and len(p) > 3: # Simple heuristic for tags
                                    stats["tags"][p] = stats["tags"].get(p, 0) + 1
        
        # Mask stats (AD 2 only or standard)
        mask_dirs = [cat / "test_public" / "ground_truth" / "bad", cat / "ground_truth"]
        for md in mask_dirs:
            if md.exists():
                masks = list(md.rglob("*.png"))
                for m_path in masks[:20]:
                    try:
                        with Image.open(m_path) as m_img:
                            m_arr = np.array(m_img)
                            perc = (np.sum(m_arr > 0) / m_arr.size) * 100
                            cat_stats["mask_percentage"].append(perc)
                    except: pass

        cat_stats["avg_contrast"] = np.mean(cat_stats["contrast"]) if cat_stats["contrast"] else 0
        cat_stats["avg_mask"] = np.mean(cat_stats["mask_percentage"]) if cat_stats["mask_percentage"] else 0
        stats["categories"][cat.name] = cat_stats
        stats["total_images"] += cat_stats["train_good"] + cat_stats["test_total"]
        
    # Aggregate stats
    stats["avg_mean"] = np.mean(stats["mean_pixel_value"], axis=0).tolist() if stats["mean_pixel_value"] else None
    stats["avg_std"] = np.mean(stats["std_pixel_value"], axis=0).tolist() if stats["std_pixel_value"] else None
    stats["resolutions"] = list(stats["resolutions"])
    
    return stats

def compare_stats(stats1, stats2, name1, name2):
    print(f"Comparison: {name1} vs {name2}")
    print("="*50)
    print(f"{'Metric':<25} | {name1:<15} | {name2:<15}")
    print("-" * 60)
    print(f"{'Total Categories':<25} | {len(stats1['categories']):<15} | {len(stats2['categories']):<15}")
    print(f"{'Total Images':<25} | {stats1['total_images']:<15} | {stats2['total_images']:<15}")
    print(f"{'Total Train (Good)':<25} | {stats1['total_train']:<15} | {stats2['total_train']:<15}")
    print(f"{'Total Test':<25} | {stats1['total_test']:<15} | {stats2['total_test']:<15}")
    print(f"{'Resolutions':<25} | {str(stats1['resolutions'][:2]):<15} | {str(stats2['resolutions'][:2]):<15}")
    
    def format_list(l):
        if l is None: return "N/A"
        if isinstance(l, list):
            return "[" + ", ".join([f"{x:.3f}" for x in l]) + "]"
        return f"{l:.3f}"

    print(f"{'Avg Pixel Mean':<25} | {format_list(stats1['avg_mean']):<15} | {format_list(stats2['avg_mean']):<15}")
    print(f"{'Avg Pixel Std':<25} | {format_list(stats1['avg_std']):<15} | {format_list(stats2['avg_std']):<15}")
    
    contrast1 = np.mean([c["avg_contrast"] for c in stats1["categories"].values()])
    contrast2 = np.mean([c["avg_contrast"] for c in stats2["categories"].values()])
    print(f"{'Avg RMS Contrast':<25} | {contrast1:<15.3f} | {contrast2:<15.3f}")
    
    mask1 = np.mean([c["avg_mask"] for c in stats1["categories"].values() if c["avg_mask"] > 0])
    mask2 = np.mean([c["avg_mask"] for c in stats2["categories"].values() if c["avg_mask"] > 0])
    print(f"{'Avg Mask Area %':<25} | {mask1:<15.2f} | {mask2:<15.2f}")

    print(f"\nRobustness Tags Found (First 10):")
    tags1 = sorted(stats1["tags"].items(), key=lambda x: x[1], reverse=True)[:10]
    tags2 = sorted(stats2["tags"].items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"{name1:<25}: {tags1}")
    print(f"{name2:<25}: {tags2}")
    
    print("\nCategory Breakdown:")
    print("-" * 100)
    print(f"{'Category':<15} | {'Train':<7} | {'Test':<7} | {'Contrast':<10} | {'Mask %':<8}")
    print("-" * 100)
    print(f"--- {name1} ---")
    for cat, s in stats1['categories'].items():
        print(f"{cat:<15} | {s['train_good']:<7} | {s['test_total']:<7} | {s['avg_contrast']:<10.3f} | {s['avg_mask']:<8.2f}")
    
    print(f"\n--- {name2} ---")
    for cat, s in stats2['categories'].items():
        print(f"{cat:<15} | {s['train_good']:<7} | {s['test_total']:<7} | {s['avg_contrast']:<10.3f} | {s['avg_mask']:<8.2f}")

if __name__ == "__main__":
    path1 = r"C:\Users\Peter\Desktop\stuff\MIUN\Research\robustanomaly\mvtec_ad_2"
    path2 = r"c:\Users\Peter\Desktop\stuff\MIUN\Semester_2\image_analysis\Project\mvtec_ad"
    
    print("Analyzing MVTec AD 2...", flush=True)
    stats1 = get_dataset_stats(path1)
    print("Analyzing MVTec AD (local)...", flush=True)
    stats2 = get_dataset_stats(path2)
    
    compare_stats(stats1, stats2, "MVTec AD 2", "MVTec AD 1")
    print("\nAnalysis complete.", flush=True)
