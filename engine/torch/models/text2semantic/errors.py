class InvalidInputError(ValueError):
    """The request cannot be served as given: bad text, references or length.

    The API server answers these with 400; every other failure is a 500.
    """
