import sys
from bing_image_downloader import downloader
import os

# Define the species and stages
species = ['oak', 'maple', 'birch']
stages = {
    'bud_emergence': 'budding emergence leaf close up',
    'expansion_maturity': 'mature green leaf summer',
    'senescence': 'autumn leaf changing color',
    'abscission': 'fallen brown leaf ground dead'
}

DATA_DIR = 'data'
IMAGES_PER_CLASS = 50

print("Downloading dataset...")

for sp in species:
    for stage_name, stage_query in stages.items():
        query = f"{sp} tree {stage_query}"
        
        # The downloader creates a folder named after the query. 
        # We need to output directly to the correct structured folder.
        output_dir = os.path.join(DATA_DIR, sp)
        
        print(f"Downloading: {sp} - {stage_name} ({query})")
        try:
            downloader.download(
                query, 
                limit=IMAGES_PER_CLASS, 
                output_dir=output_dir, 
                adult_filter_off=False, 
                force_replace=False, 
                timeout=60, 
                verbose=False
            )
            
            # The downloader saves them in data/{species}/{query}/
            # We need to rename that downloaded folder to the proper {stage_name}
            downloaded_folder = os.path.join(output_dir, query)
            target_folder = os.path.join(output_dir, stage_name)
            
            if os.path.exists(downloaded_folder):
                # If target already exists, we might need to merge or just replace.
                # Since force_replace=False above, it will append if run again.
                if os.path.exists(target_folder):
                    # Move files from downloaded to target
                    import shutil
                    for file in os.listdir(downloaded_folder):
                        shutil.move(os.path.join(downloaded_folder, file), os.path.join(target_folder, file))
                    os.rmdir(downloaded_folder)
                else:
                    os.rename(downloaded_folder, target_folder)
                
        except Exception as e:
            print(f"Failed to download {query}: {e}")

print(f"\n✅ Download complete! Images are structured in {DATA_DIR}/ directory.")
print("You can now run 'python tree_lifecycle_classifier.py' to train the model!")
