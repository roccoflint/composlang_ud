import pickle
from pathlib import Path
from abc import abstractmethod

from sqlitedict import SqliteDict


class CacheWrapper:
    def __init__(self, path: Path, flag=None) -> None:
        # filename included for compatibility with SqliteDict
        self.filename = self.path = Path(path)

    @abstractmethod
    def commit(self):
        raise NotImplementedError

    def close(self):
        ...


class PickleCache(CacheWrapper):
    d = None

    def __init__(self, path: Path, flag=None) -> None:
        super().__init__(path, flag)
        self.d = dict()

    def __getitem__(self, key):
        return self.d[key]

    def __setitem__(self, key, value):
        self.d[key] = value

    def commit(self):
        with self.path.open("wb") as f:
            pickle.dump(self.d, f)


# MRO is SqliteDict first, CacheWrapper next
class SQLiteCache(SqliteDict, CacheWrapper):
    ...
