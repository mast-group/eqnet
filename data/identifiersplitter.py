def split_camelcase(camel_case_identifier):
    """
    :type camel_case_identifier: str
    :param camel_case_identifier:
    :return:
    """
    if not len(camel_case_identifier):
        return []

    # split into words based on adjacent cases being the same
    result = []
    current = str(camel_case_identifier[0])
    prev_upper = camel_case_identifier[0].isupper()
    prev_digit = camel_case_identifier[0].isdigit()
    prev_special = not camel_case_identifier[0].isalnum()
    for c in camel_case_identifier[1:]:
        upper = c.isupper()
        digit = c.isdigit()
        special = not c.isalnum()
        new_upper_word = upper and not prev_upper
        new_digit_word = digit and not prev_digit
        new_special_word = special and not prev_special
        if new_digit_word or new_upper_word or new_special_word:
            result.append(current)
            current = c
        elif not upper and prev_upper and len(current) > 1:
            result.append(current[:-1])
            current = current[-1] + c
        elif not digit and prev_digit:
            result.append(current)
            current = c
        elif not special and prev_special:
            result.append(current)
            current = c
        else:
            current += c
        prev_digit = digit
        prev_upper = upper
        prev_special = special
    result.append(current)
    return result


def split_identifier_into_parts(identifier):
    """
    Split a single identifier into parts.
    :param identifier:
    :return:
    """
    if identifier is None:
        return [None]
    snake_case = identifier.split("_")

    identifier_parts = []
    for i in range(len(snake_case)):
        part = snake_case[i]
        if len(part) > 0:
            identifier_parts.extend(split_camelcase(part))
        if i < len(snake_case) - 1:
            identifier_parts.append("_")
    return identifier_parts
