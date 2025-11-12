from PyQt5.QtCore import QObject, pyqtSignal
from .update_checker import check_for_updates

class UpdateWorker(QObject):
    """
    A worker that checks for updates in a background thread.
    """
    finished = pyqtSignal(object, str)  # Emits update_info and error_message

    def run(self):
        """
        Executes the update check and emits the result.
        """
        update_info = None
        error_message = None
        try:
            update_info = check_for_updates()
        except Exception as e:
            error_message = str(e)
        
        self.finished.emit(update_info, error_message)