from logging import Logger


def log_no_split_dict(logger: Logger):
    message = f"No split_dict is specified"
    logger.warn(message)

def log_not_found_split_dict(logger: Logger, split_dict_path, e=None):
    message = f'Cannot read split_dict from "{split_dict_path}"' + (
        f"due to e={e}" if e else ""
    )

    logger.warn(message)


def log_not_found_split_dict_key(logger: Logger, key, e=None):
    message = f"split_dict key={key} not found."
    logger.warn(message)


def log_not_found_label(logger: Logger, image_id, image_path=None):
    message = (
        f"Image id={image_id}"
        + (f"(path={image_path})" if image_path else "")
        + " have no label file"
    )
    logger.warn(message)
