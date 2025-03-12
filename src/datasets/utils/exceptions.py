class SplitDictKeyException(RuntimeError):
    def __init__(self, split) -> None:
        super().__init__(f"Invalid split_dict: split={split} not found")

