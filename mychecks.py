from functools import wraps


def expect_columns(input, output):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if input is not None:
                df = args[0]
                column_names = set([c for c in df.head()])
                for column_name in input:
                    if column_name not in column_names:
                        raise ValueError(
                            f"this function was provided a dataframe which does not respect the contract :  \
                            column {column_name} not found in input")  # noqa E501
            df = func(*args, **kwargs)
            if output is not None:
                column_names = [df.index.name] + [c for c in df.head()]
                if output is not None:
                    for column_name in output:
                        if column_name not in column_names:
                            raise ValueError(
                                f"this function outputs a dataframe which does not respect the contract : \
                                column {column_name} not found in output")  # noqa E501
            return df
        return wrapper
    return decorator
