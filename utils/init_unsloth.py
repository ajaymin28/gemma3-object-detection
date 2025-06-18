from utils.logman import logger
FastModel = None
# Optional – comment below imports if you are not planinng to use unsloth
try: from unsloth import FastModel
except ImportError as e: logger.warning(f"Unsloth import error : {e}")
except NotImplementedError as e: logger.warning(f"Unsloth NotImplementedError error : {e}")