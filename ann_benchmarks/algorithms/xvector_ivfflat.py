import psycopg
import subprocess
import sys
from datetime import datetime


from psycopg.types import TypeInfo
from psycopg.adapt import Loader, Dumper
from psycopg.pq import Format
import numpy as np
from struct import pack, unpack
from .base import BaseANN


class XVectorIvfflat(BaseANN):
    def __init__(self, metric, lists):
        self._metric = metric
        self._cur = None
        self._lists = lists

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")


    @staticmethod
    def from_db(value):
        if value is None or isinstance(value, np.ndarray):
            return value
        return np.array(value[1:-1].split(','), dtype=np.float32)

    @staticmethod
    def from_db_binary(value):
        if value is None:
            return value

        (dim, unused) = unpack('>HH', value[:4])
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

        return '[' + ','.join([str(float(v)) for v in value]) + ']'



    @staticmethod
    def to_db_binary(value):
        if value is None:
            return value

        value = np.asarray(value, dtype='>f')

        if value.ndim != 1:
            raise ValueError('expected ndim to be 1')

        return pack('>HH', value.shape[0], 0) + value.tobytes()



    class VectorDumper(Dumper):
        format = Format.TEXT
        def dump(self, obj):
            return XVectorIvfflat.to_db(obj).encode("utf8")

    class VectorBinaryDumper(VectorDumper):
        format = Format.BINARY
        def dump(self, obj):
            return XVectorIvfflat.to_db_binary(obj)

    class VectorLoader(Loader):
        format = Format.TEXT
        def load(self, data):
            if isinstance(data, memoryview):
                data = bytes(data)
            return XVectorIvfflat.from_db(data.decode("utf8"))

    class VectorBinaryLoader(VectorLoader):
        format = Format.BINARY
        def load(self, data):
            if isinstance(data, memoryview):
                data = bytes(data)
            return XVectorIvfflat.from_db_binary(data)

    def register_vector_info(self, context, info):
        if info is None:
            raise psycopg.ProgrammingError('xvector type not found in the database')
        info.register(context)

        text_dumper = type('', (XVectorIvfflat.VectorDumper,), {'oid': info.oid})
        binary_dumper = type('', (XVectorIvfflat.VectorBinaryDumper,), {'oid': info.oid})

        adapters = context.adapters
        adapters.register_dumper('numpy.ndarray', text_dumper)
        adapters.register_dumper('numpy.ndarray', binary_dumper)
        adapters.register_loader(info.oid, XVectorIvfflat.VectorLoader)
        adapters.register_loader(info.oid, XVectorIvfflat.VectorBinaryLoader)

    def register_vector(self, context):
        info = TypeInfo.fetch(context, 'xvector')
        self.register_vector_info(context, info)


    def fit(self, X):
        #subprocess.run("service postgresql start", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        conn = psycopg.connect(user="ann", password="ann", dbname="ann", host="localhost")
        self.register_vector(conn)
        dim=X.shape[1]
        cur = conn.cursor()
        cur.execute("CREATE TABLE items (id int, embedding xvector(%d))" % X.shape[1])
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        print("copying data...")
        with cur.copy("COPY items (id, embedding) FROM STDIN") as copy:
            for i, embedding in enumerate(X):
                copy.write_row((i, embedding))
        print("creating index...")
        if self._metric == "angular":
            cur.execute(
                "CREATE INDEX ON items USING ivfflat (embedding ivfflat_cosine_ops) WITH (lists = %d)" % self._lists
            )
        elif self._metric == "euclidean":
            cur.execute("CREATE INDEX ON items USING ivfflat (embedding ivfflat_l2_ops) WITH (lists = %d)" % self._lists)
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("done!")
        self._cur = cur

    def set_query_arguments(self, probes):
        self._probes = probes
        self._cur.execute("SET ivfflat.probes = %d" % probes)
        # TODO set based on available memory
        self._cur.execute("SET work_mem = '256MB'")
        # disable parallel query execution
        self._cur.execute("SET max_parallel_workers_per_gather = 0")

    def query(self, v, n):
        self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def __str__(self):
        return f"XVector(lists={self._lists}, probes={self._probes})"
