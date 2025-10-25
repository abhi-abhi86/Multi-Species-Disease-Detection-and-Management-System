                                        
import os
import shutil
import re


def prepare_dataset():
    """
    Automates the process of creating a training dataset by organizing images
    from the 'diseases' directory into a 'dataset' directory, which is what the
    training script expects.
    """
                                                      
                                                       
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(base_dir, 'diseases')
    dest_dir = os.path.join(base_dir, 'dataset')

    print("--- Starting Dataset Preparation ---")

                                                         
    if not os.path.exists(source_dir):
        print(f"ERROR: The source directory '{source_dir}' was not found.")
        print("Please ensure you have the 'diseases' directory with image subfolders.")
        return

                                                                                       
    if os.path.exists(dest_dir):
        print(f"Removing existing dataset directory: '{dest_dir}'")
        shutil.rmtree(dest_dir)
    print(f"Creating new dataset directory: '{dest_dir}'")
    os.makedirs(dest_dir)

                                                                       
    image_count = 0
    class_count = 0
    for root, dirs, files in os.walk(source_dir):
                                                           
        if os.path.basename(root) == 'images':
                                                                             
                                                                              
            class_name = os.path.basename(os.path.dirname(root))

                                                            
            safe_class_name = re.sub(r'[\s/\\:*?"<>|]+', '_', class_name).lower()

            if not safe_class_name:
                continue

                                                           
            class_dest_dir = os.path.join(dest_dir, safe_class_name)
            if not os.path.exists(class_dest_dir):
                print(f"Creating class folder: '{safe_class_name}'")
                os.makedirs(class_dest_dir)
                class_count += 1

                                         
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    source_path = os.path.join(root, file)
                    dest_path = os.path.join(class_dest_dir, file)
                    shutil.copy(source_path, dest_path)
                    image_count += 1

    print("\n--- Dataset Preparation Complete ---")
    if image_count > 0:
        print(f"Successfully copied {image_count} images into {class_count} class folders.")
        print(f"Your dataset is now ready for training at: '{dest_dir}'")
        print("You can now run 'train_disease_classifier.py'.")
    else:
        print("WARNING: No images were found in any 'images' subdirectories within the 'diseases' folder.")
        print("Please check your directory structure.")


if __name__ == "__main__":
    prepare_dataset()
