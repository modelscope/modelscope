def dec(param1):
    print(param1)

    def in_dec(func):
        def in_func(name):
            return func(name)
        return in_func
    return in_dec


@dec("dec1")
def aa(param):
    print(param)
    return

aa("heell")