import tiktoken

def num_tokens_from_string(string: str ) -> int:
    """Returns the number of tokens in a text string."""
    encoding_name:str = "cl100k_base"
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
