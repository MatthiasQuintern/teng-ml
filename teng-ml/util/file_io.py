from os import listdir, path

def add_zeros(v: int, digits=3):
    """
    return v as string, add leading zeros if len(str(v)) < digits
    """
    s = str(v)
    return '0' * (max(digits - len(s), 0)) + s


def get_next_digits(basename, directory=".", digits=3):
    """
    get the next filename digits
    example:
        basename = file
        directory has file001.csv, file002.pkl, file004.csv
        -> return 005
    """
    files = listdir(directory)
    files.sort()
    files.reverse()
    lowest_number = -1
    for file in files:
        if not file.startswith(basename): continue
        try:
            dot = file.rfind('.')
            if dot > 0: file = file[:dot]
            number = int(file.replace(basename, ""))
            if number < lowest_number: continue
            lowest_number = number
        except ValueError:
            continue

    return add_zeros(lowest_number+1)
