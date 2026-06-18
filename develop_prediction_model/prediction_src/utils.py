# Contains utils functions for finding all important files.

import logging
import os
from datetime import datetime

# Define helper function
def find_and_remove_first_element(strings:list, search_term:str) -> str:
    """ Returns the first element that starts with the search term and removes it from the list.

    Args:
        strings (list): List of strings.
        search_term (str): The search term that the element should start with.

    Returns:
        str: The found element or None if no such element is found.
    """

    for index, string in enumerate(strings):
        if string.startswith(search_term):
            return strings.pop(index)
    return None


def find_files(pwd: str) -> dict:
    """Find all data files in one directory. Should not be used in practice, but I am to lazy to specify the files every time.
    TODO: At later versions, this should be hardly specified in a configuration file!

    Args:
        pwd (str): path to directory containing the files ONLY ONCE.

    Returns:
        dict: directory with all paths.
    """
    
    file_list = os.listdir(pwd)
    return {
        "participant_filled_questionnaire_file": pwd + "/"+ find_and_remove_first_element(file_list, "participant_filled_questionnaire"),
        "participant_file": pwd + "/" +  find_and_remove_first_element(file_list, "participant"),
        "answer_question_answer_file": pwd + "/" + find_and_remove_first_element(file_list, "answer_question_answer"),
        "answer_file": pwd + "/" + find_and_remove_first_element(file_list, "answer"),
    }


def create_folder(path: str) -> str:
    """ Create a new folder in the given path."""
    if not(os.path.exists(path)):
        if "/" in path:
            _ = create_folder("/".join(path.split("/")[:-1]))
        os.mkdir(path)
        logging.debug(f"{path} created.")
    return path


def create_logger(process_id, log_path, exp_name):
    """Creates a logger with nice information and time stamps.

    Returns:
        logging.logger: _description_
    """

    import matplotlib

    matplotlib_logger = logging.getLogger('matplotlib')
    matplotlib_logger.setLevel(logging.WARNING)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    #process_id = os.getpid()
    now = datetime.now()
    log_file = "{}/{}-{}-{}.log".format(log_path, now.strftime("%Y%m%d-%H%M%S"), exp_name, process_id)
    log_format = "%(asctime)s-%(process)d - %(levelname)s - %(message)s"

    # Create Logger
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format=log_format)
    
    logger = logging.getLogger("Analysis")

    handler = logging.StreamHandler()
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info(f'Process ID: {process_id}')
    logger.info(f'Experiment {exp_name} started')
    return logger
