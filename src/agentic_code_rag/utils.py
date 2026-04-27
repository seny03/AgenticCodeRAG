import re


def camel_case_to_snake(input_string, separator='_'):
    """
    Convert a camel case string into a snake case one.
    (The original string is returned if is not a valid camel case string)

    Example:
    >>> camel_case_to_snake('ThisIsACamelStringTest')  # returns 'this_is_a_camel_string_test'

    :param input_string: String to convert.
    :type input_string: str
    :param separator: Sign to use as separator.
    :type separator: str
    :return: Converted string.
    """
    if not isinstance(input_string, str):
        return input_string
    
    # Check if it's already snake case or contains no uppercase letters
    if separator in input_string or not any(c.isupper() for c in input_string):
        return input_string
    
    # Handle consecutive uppercase letters (like 'HTTPRequest' -> 'http_request')
    # First, insert separator between uppercase letters followed by lowercase
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1' + separator + r'\2', input_string)
    
    # Then insert separator between lowercase/numbers and uppercase
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1' + separator + r'\2', s1)
    
    # Convert to lowercase
    return s2.lower()


if __name__ == "__main__":
    # Test cases
    test_cases = [
        ('ThisIsACamelStringTest', 'this_is_a_camel_string_test'),
        ('HTTPRequest', 'http_request'),
        ('XMLHttpRequest', 'xml_http_request'),
        ('already_snake_case', 'already_snake_case'),
        ('lowercase', 'lowercase'),
        ('', ''),
        ('Single', 'single'),
        ('multipleCAPS', 'multiple_caps'),
    ]
    
    for input_str, expected in test_cases:
