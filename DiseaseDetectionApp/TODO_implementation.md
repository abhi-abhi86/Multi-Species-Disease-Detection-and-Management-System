# Implementation TODO

1. Modify add_disease_dialog.py:
   - Add "Risk Factors" field.
   - Modify __init__ to accept prefilled data (image_path, domain).
   - Add main block for standalone execution.

2. Modify data_handler.py:
   - Update load_database to load from user_added_diseases/ as well.
   - Update save_disease to save to user_added_diseases/domain/name.json, create images/ subfolder, copy image if provided.

3. Modify train_disease_classifier.py:
   - Update prepare_dataset_for_training to include user_added_diseases/ in addition to diseases/.

4. Modify main_window.py:
   - In on_diagnosis_complete, if result name is "No Confident Match Found", add a button "Add New Disease" that opens the dialog with prefilled image_path and domain.
   - After saving disease, run train_disease_classifier.py in a thread, then reload database and MLProcessor.

5. Test standalone dialog and integrated flow.
