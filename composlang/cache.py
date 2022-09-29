import pickle, yaml
from pathlib import Path
from abc import abstractmethod

from sqlitedict import SqliteDict
from composlang.utils import log, pathify


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
        path = pathify(path)
        super().__init__(path, flag)
        if path.exists():
            with path.open("rb") as f:
                self.d = pickle.load(f)
        else:
            log(f"unable to open PickleCache from {path}")
            self.d = dict()

    def __getitem__(self, key):
        return self.d[key]

    def __setitem__(self, key, value):
        self.d[key] = value

    def commit(self):
        if self.d:
            with self.path.open("wb") as f:
                pickle.dump(self.d, f)
        else:
            log(f"skipping `commit` call on empty PickleCache to {self.path}")


class YAMLCache(CacheWrapper):
    d = None

    def __init__(self, path: Path, flag=None) -> None:
        path = pathify(path)
        super().__init__(path, flag)
        if path.exists():
            with path.open("r") as f:
                self.d = yaml.load(f, Loader=yaml.SafeLoader)
        else:
            log(f"unable to open YAMLCache from {path}")
            self.d = dict()

    def __getitem__(self, key):
        return self.d[key]

    def __setitem__(self, key, value):
        self.d[key] = value

    def commit(self):
        if self.d:
            with self.path.open("w") as f:
                yaml.dump(self.d, f, Dumper=yaml.SafeDumper)
        else:
            log(f"skipping `commit` call on empty YAMLCache to {self.path}")


# MRO is SqliteDict first, CacheWrapper next
class SQLiteCache(SqliteDict, CacheWrapper):
    ...
