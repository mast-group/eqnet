from data import identifiersplitter


class NamingData:
    @staticmethod
    def get_first_name_subtoken_from(name: str):
        parenthesis_location = name.find('(')
        method_name = name[name.rfind('.', 1, parenthesis_location) + 1:parenthesis_location]
        method_subtokens = identifiersplitter.split_identifier_into_parts(method_name)
        return method_subtokens[0].lower()

    @staticmethod
    def get_return_type_from(name: str) -> str:
        start_pos = name.rfind(':')
        assert start_pos > 0
        return name[start_pos + 1:]

    @staticmethod
    def get_nargs_from(name: str) -> str:
        args = name[name.rfind('(') + 1:name.rfind(')')]
        if len(args) == 0:
            return str(0)
        in_brace = 0
        nargs = 1
        for char in args:
            if char == ',' and in_brace == 0:
                nargs += 1
            elif char == '<':
                in_brace += 1
            elif char == '>':
                in_brace -= 1
        return str(nargs)
