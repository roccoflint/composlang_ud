
import typing
from sys import stderr
from pathlib import Path


def log(*args, **kwargs):
    '''
    placeholder logging method to be changed later
    '''
    print('** info:', *args, **kwargs, file=stderr)


def pathify(fpth: typing.Union[Path, str, typing.Any]) -> Path:
    '''
    returns a resolved `Path` object after expanding user and shorthands/symlinks
    '''
    return Path(fpth).expanduser().resolve()


def iterable_from_directory_or_filelist(directory_or_filelist) -> typing.Iterable[Path]:
    '''
    constructs an iterable over Path objects using a given directory, file, or
    list of files as either str or Path objects
    '''
    # either it must be a list of files, or a directory containing files, or a file
    # first, assume it is a directory or individual file
    try: 
        directory_or_filelist = pathify(directory_or_filelist)
        if directory_or_filelist.is_dir():
            files = [*directory_or_filelist.iterdir()]
        else:
            files = [directory_or_filelist]
    # next, assume we are given a list of filepaths as strs or or Path-like objects
    except TypeError as e:
        files = list(map(pathify, directory_or_filelist))

    return files