import psycopg
import subprocess
import sys

from psycopg.types import TypeInfo
from psycopg.adapt import Loader, Dumper
from psycopg.pq import Format
import numpy as np
from struct import pack, unpack

from .base import BaseANN


class XVector(BaseANN):
    def __init__(self, metric):
        self._metric = metric
        self._ef = 400
        self._cur = None
        self._query = "SELECT id FROM items ORDER BY vector <?> %s LIMIT %s"

    @staticmethod
    def from_db(value):
        if value is None or isinstance(value, np.ndarray):
            return value
        return np.array(value.split(','), dtype=np.float32)

    @staticmethod
    def from_db_binary(value):
        if value is None:
            return value
        (dim, extra, ds) = unpack('>LLL', value[:12])
        return np.frombuffer(value, dtype='>f', count=dim, offset=4).astype(dtype=np.float32)

    @staticmethod
    def to_db(value, dim=None):
        if value is None:
            return value
        if isinstance(value, np.ndarray):
            if value.ndim != 1:
                raise ValueError('expected ndim to be 1')
            if not np.issubdtype(value.dtype, np.integer) and not np.issubdtype(value.dtype, np.floating):
                raise ValueError('dtype must be numeric')
            value = value.tolist()
        if dim is not None and len(value) != dim:
            raise ValueError('expected %d dimensions, not %d' % (dim, len(value)))
        return '{' + ','.join([str(float(v)) for v in value]) + '}'

    @staticmethod
    def to_db_binary(value):
        if value is None:
            return value
        value = np.asarray(value, dtype='>f')
        if value.ndim != 1:
            raise ValueError('expected ndim to be 1')
        return pack('>LLL', value.shape[0], 0, 0) + value.tobytes()

    class VectorDumper(Dumper):
        format = Format.TEXT
        def dump(self, obj):
            return XVector.to_db(obj).encode("utf8")

    class VectorBinaryDumper(VectorDumper):
        format = Format.BINARY
        def dump(self, obj):
            return XVector.to_db_binary(obj)

    class VectorLoader(Loader):
        format = Format.TEXT
        def load(self, data):
            if isinstance(data, memoryview):
                data = bytes(data)
            return XVector.from_db(data.decode("utf8"))

    class VectorBinaryLoader(VectorLoader):
        format = Format.BINARY
        def load(self, data):
            if isinstance(data, memoryview):
                data = bytes(data)
            return XVector.from_db_binary(data)

    def register_vector_info(self, context, info):
        if info is None:
            raise psycopg.ProgrammingError('xvector type not found in the database')
        info.register(context)

        text_dumper = type('', (XVector.VectorDumper,), {'oid': info.oid})
        binary_dumper = type('', (XVector.VectorBinaryDumper,), {'oid': info.oid})

        adapters = context.adapters
        adapters.register_dumper('numpy.ndarray', text_dumper)
        adapters.register_dumper('numpy.ndarray', binary_dumper)
        adapters.register_loader(info.oid, XVector.VectorLoader)
        adapters.register_loader(info.oid, XVector.VectorBinaryLoader)

    def register_vector(self, context):
        info = TypeInfo.fetch(context, 'xvector')
        self.register_vector_info(context, info)

    def fit(self, X):
        subprocess.run("service postgresql start", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        conn = psycopg.connect(user="ann", password="ann", dbname="ann", host="localhost")
        self.register_vector(conn)

        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS items (id int, vector float4[])")
        cur.execute("ALTER TABLE items ALTER COLUMN vector SET STORAGE PLAIN")
        print("Copying vector data...")
        # Building index later is faster 
        with cur.copy("COPY items (id, vector) FROM STDIN") as copy:
            for i, embedding in enumerate(X):
                copy.write_row((i, embedding))        
        print("Building index...")
        dim=X.shape[1]
        cur.execute(f"CREATE INDEX IF NOT EXISTS items_hnsw_idx ON items USING xvector_hnsw(vector) WITH(dim={dim})")
        self._cur = cur

    def set_query_arguments(self, ef):
        self._cur.execute("SET work_mem = '256MB'")
        self._cur.execute("SET max_parallel_workers_per_gather = 0")
        self._ef = ef

    def query(self, v, n):
        self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def __str__(self):
        return "XVector"
