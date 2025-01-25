from enum import Enum
import tiktoken
from typing import Callable, Optional

class Language(str, Enum):
    """Supported languages for language-specific chunking."""
    CPP = "cpp"
    GO = "go"
    JAVA = "java"
    KOTLIN = "kotlin"
    JS = "js"
    TS = "ts"
    PHP = "php"
    PROTO = "proto"
    PYTHON = "python"
    RST = "rst"
    RUBY = "ruby"
    RUST = "rust"
    SCALA = "scala"
    SWIFT = "swift"
    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"
    SOL = "sol"
    CSHARP = "csharp"
    COBOL = "cobol"
    C = "c"
    LUA = "lua"
    PERL = "perl"

def get_token_count(
    string: str,
    encoding_name: str = "cl100k_base",
    model_name: Optional[str] = None,
    **kwargs
) -> int:
    """
    Count the number of tokens in a string using tiktoken.
    
    Args:
        string: The text to count tokens for
        encoding_name: The name of the tiktoken encoding to use
        model_name: Optional model name to use specific encoding
        **kwargs: Additional arguments passed to tiktoken encoder
        
    Returns:
        Number of tokens in the string
    """
    try:
        if model_name:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)
            
        allowed_special = kwargs.get('allowed_special', set())
        disallowed_special = kwargs.get('disallowed_special', 'all')
        
        return len(enc.encode(
            string,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special
        ))
    except Exception as e:
        raise ValueError(f"Error counting tokens: {str(e)}")

def get_character_count(text: str) -> int:
    """
    Count the number of characters in a string.
    
    Args:
        text: The text to count characters for
        
    Returns:
        Number of characters in the string
    """
    return len(text)

def get_length_function(length_type: str = "token", **kwargs) -> Callable[[str], int]:
    """
    Get a length function based on the specified type.
    
    Args:
        length_type: Type of length function ('token' or 'character')
        **kwargs: Additional arguments passed to token counter
        
    Returns:
        A callable that takes a string and returns its length
    """
    if length_type == "token":
        return lambda x: get_token_count(x, **kwargs)
    elif length_type == "character":
        return get_character_count
    else:
        raise ValueError(
            f"Unknown length type: {length_type}. "
            "Choose 'token' or 'character'"
        )