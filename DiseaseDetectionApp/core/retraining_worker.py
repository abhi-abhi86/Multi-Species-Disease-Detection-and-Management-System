
import os
import sys
from PySide6.QtCore import QThread, Signal


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class RetrainingWorker(QThread):
    """
    Worker thread for running model retraining in the background.
    Emits signals for completion or error.
    """
    finished = Signal()
    error = Signal(str)

    def run(self):
        """Run the retraining process."""
        try:

            import subprocess
            import sys


            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            train_script = os.path.join(base_dir, 'train_disease_classifier.py')


            result = subprocess.run([sys.executable, train_script],
                                  capture_output=True, text=True, cwd=base_dir)

            if result.returncode == 0:
                self.finished.emit()
            else:
                error_msg = f"Training failed with return code {result.returncode}\n"
                error_msg += f"STDOUT: {result.stdout}\n"
                error_msg += f"STDERR: {result.stderr}"
                self.error.emit(error_msg)

        except Exception as e:
            self.error.emit(f"Exception during retraining: {str(e)}")
