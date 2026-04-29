# src/check_writer_contents.py
import os

ROOT = r"E:\handwriting_matcher\data"
writers = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT,d))])
print("Total writer folders found:", len(writers))
print()

# Inspect first 8 writers (you can change the slice)
for w in writers[:8]:
    wpath = os.path.join(ROOT, w)
    items = os.listdir(wpath)
    files_direct = [f for f in items if os.path.isfile(os.path.join(wpath, f))]
    dirs_direct  = [d for d in items if os.path.isdir(os.path.join(wpath, d))]
    # Count images directly inside
    img_count_direct = sum(1 for f in files_direct if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff','.bmp')))
    # Count images inside immediate subfolders (one level)
    img_count_in_subs = 0
    subs_detail = []
    for d in dirs_direct:
        subpath = os.path.join(wpath, d)
        subfiles = [f for f in os.listdir(subpath) if os.path.isfile(os.path.join(subpath,f))]
        sub_img_count = sum(1 for f in subfiles if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff','.bmp')))
        if sub_img_count>0:
            subs_detail.append((d, sub_img_count))
            img_count_in_subs += sub_img_count

    print("Writer:", w)
    print("  Direct files count:", len(files_direct), "  image files directly:", img_count_direct)
    print("  Direct subfolders count:", len(dirs_direct))
    if subs_detail:
        print("  Subfolders with images (name, image_count):", subs_detail[:8])
    print("  Total images found (direct + immediate subfolders):", img_count_direct + img_count_in_subs)
    print()
