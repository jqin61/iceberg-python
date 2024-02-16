# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC, PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import BucketTransform, TruncateTransform
from pyiceberg.types import (
    IntegerType,
    NestedField,
    StringType,
    StructType,
)


def test_partition_field_init() -> None:
    bucket_transform = BucketTransform(100)  # type: ignore
    partition_field = PartitionField(3, 1000, bucket_transform, "id")

    assert partition_field.source_id == 3
    assert partition_field.field_id == 1000
    assert partition_field.transform == bucket_transform
    assert partition_field.name == "id"
    assert partition_field == partition_field
    assert str(partition_field) == "1000: id: bucket[100](3)"
    assert (
        repr(partition_field)
        == "PartitionField(source_id=3, field_id=1000, transform=BucketTransform(num_buckets=100), name='id')"
    )


def test_unpartitioned_partition_spec_repr() -> None:
    assert repr(PartitionSpec()) == "PartitionSpec(spec_id=0)"


def test_partition_spec_init() -> None:
    bucket_transform: BucketTransform = BucketTransform(4)  # type: ignore

    id_field1 = PartitionField(3, 1001, bucket_transform, "id")
    partition_spec1 = PartitionSpec(id_field1)

    assert partition_spec1.spec_id == 0
    assert partition_spec1 == partition_spec1
    assert partition_spec1 != id_field1
    assert str(partition_spec1) == f"[\n  {str(id_field1)}\n]"
    assert not partition_spec1.is_unpartitioned()
    # only differ by PartitionField field_id
    id_field2 = PartitionField(3, 1002, bucket_transform, "id")
    partition_spec2 = PartitionSpec(id_field2)
    assert partition_spec1 != partition_spec2
    assert partition_spec1.compatible_with(partition_spec2)
    assert partition_spec1.fields_by_source_id(3) == [id_field1]
    # Does not exist
    assert partition_spec1.fields_by_source_id(1925) == []


def test_partition_compatible_with() -> None:
    bucket_transform: BucketTransform = BucketTransform(4)  # type: ignore
    field1 = PartitionField(3, 100, bucket_transform, "id")
    field2 = PartitionField(3, 102, bucket_transform, "id")
    lhs = PartitionSpec(
        field1,
    )
    rhs = PartitionSpec(field1, field2)
    assert not lhs.compatible_with(rhs)


def test_unpartitioned() -> None:
    assert len(UNPARTITIONED_PARTITION_SPEC.fields) == 0
    assert UNPARTITIONED_PARTITION_SPEC.is_unpartitioned()
    assert str(UNPARTITIONED_PARTITION_SPEC) == "[]"


def test_serialize_unpartitioned_spec() -> None:
    assert UNPARTITIONED_PARTITION_SPEC.model_dump_json() == """{"spec-id":0,"fields":[]}"""


def test_serialize_partition_spec() -> None:
    partitioned = PartitionSpec(
        PartitionField(source_id=1, field_id=1000, transform=TruncateTransform(width=19), name="str_truncate"),
        PartitionField(source_id=2, field_id=1001, transform=BucketTransform(num_buckets=25), name="int_bucket"),
        spec_id=3,
    )
    assert (
        partitioned.model_dump_json()
        == """{"spec-id":3,"fields":[{"source-id":1,"field-id":1000,"transform":"truncate[19]","name":"str_truncate"},{"source-id":2,"field-id":1001,"transform":"bucket[25]","name":"int_bucket"}]}"""
    )


def test_deserialize_unpartition_spec() -> None:
    json_partition_spec = """{"spec-id":0,"fields":[]}"""
    spec = PartitionSpec.model_validate_json(json_partition_spec)

    assert spec == PartitionSpec(spec_id=0)


def test_deserialize_partition_spec() -> None:
    json_partition_spec = """{"spec-id": 3, "fields": [{"source-id": 1, "field-id": 1000, "transform": "truncate[19]", "name": "str_truncate"}, {"source-id": 2, "field-id": 1001, "transform": "bucket[25]", "name": "int_bucket"}]}"""

    spec = PartitionSpec.model_validate_json(json_partition_spec)

    assert spec == PartitionSpec(
        PartitionField(source_id=1, field_id=1000, transform=TruncateTransform(width=19), name="str_truncate"),
        PartitionField(source_id=2, field_id=1001, transform=BucketTransform(num_buckets=25), name="int_bucket"),
        spec_id=3,
    )


def test_partition_type(table_schema_simple: Schema) -> None:
    spec = PartitionSpec(
        PartitionField(source_id=1, field_id=1000, transform=TruncateTransform(width=19), name="str_truncate"),
        PartitionField(source_id=2, field_id=1001, transform=BucketTransform(num_buckets=25), name="int_bucket"),
        spec_id=3,
    )

    assert spec.partition_type(table_schema_simple) == StructType(
        NestedField(field_id=1000, name="str_truncate", field_type=StringType(), required=False),
        NestedField(field_id=1001, name="int_bucket", field_type=IntegerType(), required=False),
    )

#############################
####TESTING PARTITION KEY####
#############################

from datetime import date, datetime
import uuid
import pytz

TEST_DATA_WITH_NULL = {
    'bool': [False, None, True],
    'string': ['a', None, 'z'],
    # Go over the 16 bytes to kick in truncation
    'string_long': ['a' * 22, None, 'z' * 22],
    'int': [1, None, 9],
    'long': [1, None, 9],
    'float': [0.0, None, 0.9],
    'double': [0.0, None, 0.9],
    'timestamp': [datetime(2023, 1, 1, 19, 25, 00), None, datetime(2023, 3, 1, 19, 25, 00)],
    'timestamptz': [
        datetime(2023, 1, 1, 19, 25, 00, tzinfo=pytz.timezone('America/New_York')),
        None,
        datetime(2023, 3, 1, 19, 25, 00, tzinfo=pytz.timezone('America/New_York')),
    ],
    'date': [date(2023, 1, 1), None, date(2023, 3, 1)],
    # Not supported by Spark
    # 'time': [time(1, 22, 0), None, time(19, 25, 0)],
    # Not natively supported by Arrow
    # 'uuid': [uuid.UUID('00000000-0000-0000-0000-000000000000').bytes, None, uuid.UUID('11111111-1111-1111-1111-111111111111').bytes],
    'binary': [b'\01', None, b'\22'],
    'fixed': [
        uuid.UUID('00000000-0000-0000-0000-000000000000').bytes,
        None,
        uuid.UUID('11111111-1111-1111-1111-111111111111').bytes,
    ],
}
import pyarrow as pa
import pytest

@pytest.fixture(scope="session")
def arrow_table_with_null() -> pa.Table:
    """PyArrow table with all kinds of columns"""
    pa_schema = pa.schema([
        ("bool", pa.bool_()),
        ("string", pa.string()),
        ("string_long", pa.string()),
        ("int", pa.int32()),
        ("long", pa.int64()),
        ("float", pa.float32()),
        ("double", pa.float64()),
        ("timestamp", pa.timestamp(unit="us")),
        ("timestamptz", pa.timestamp(unit="us", tz="UTC")),
        ("date", pa.date32()),
        # Not supported by Spark
        # ("time", pa.time64("us")),
        # Not natively supported by Arrow
        # ("uuid", pa.fixed(16)),
        ("binary", pa.binary()),
        ("fixed", pa.binary(16)),
    ])
    return pa.Table.from_pydict(TEST_DATA_WITH_NULL, schema=pa_schema)


from pyiceberg.schema import Schema
from pyiceberg.transforms import IdentityTransform, MonthTransform, YearTransform, DayTransform, HourTransform, TruncateTransform, BucketTransform, IdentityTransform
from pyiceberg.types import (
    BinaryType,
    BooleanType,
    DateType,
    DoubleType,
    FixedType,
    FloatType,
    IntegerType,
    LongType,
    NestedField,
    StringType,
    TimestampType,
    TimestamptzType,
)

TABLE_SCHEMA = Schema(
    NestedField(field_id=1, name="bool", field_type=BooleanType(), required=False),
    NestedField(field_id=2, name="string", field_type=StringType(), required=False),
    NestedField(field_id=3, name="string_long", field_type=StringType(), required=False),
    NestedField(field_id=4, name="int", field_type=IntegerType(), required=False),
    NestedField(field_id=5, name="long", field_type=LongType(), required=False),
    NestedField(field_id=6, name="float", field_type=FloatType(), required=False),
    NestedField(field_id=7, name="double", field_type=DoubleType(), required=False),
    NestedField(field_id=8, name="timestamp", field_type=TimestampType(), required=False),
    NestedField(field_id=9, name="timestamptz", field_type=TimestamptzType(), required=False),
    NestedField(field_id=10, name="date", field_type=DateType(), required=False),
    # NestedField(field_id=11, name="time", field_type=TimeType(), required=False),
    # NestedField(field_id=12, name="uuid", field_type=UuidType(), required=False),
    NestedField(field_id=11, name="binary", field_type=BinaryType(), required=False),
    NestedField(field_id=12, name="fixed", field_type=FixedType(16), required=False),
)

from pyiceberg.typedef import  Record
from pyiceberg.partitioning import PartitionField, PartitionSpec, PartitionFieldValue, PartitionKey

@pytest.mark.mexico
@pytest.mark.parametrize(
    "partition_fields, partition_field_values, expected_partition_record, expected_hive_partition_path_slice", 
    [
        # Year Transform
        (
            [PartitionField(source_id=8, field_id=1001, transform=MonthTransform(), name="partition_field")],
            [PartitionFieldValue(source_id=8, value=datetime(2023, 1, 1, 11, 55, 59, 999999))],
            Record(partition_field=((2023-1970) * 12)),
            "partition_field=2023-01"
        ),
        (
            [PartitionField(source_id=9, field_id=1001, transform=MonthTransform(), name="partition_field")],
            [PartitionFieldValue(source_id=9, value=datetime(2023, 1, 1, 11, 55, 59,999999, tzinfo=pytz.timezone('America/New_York')))],
            Record(partition_field=((2023-1970) * 12)),
            "partition_field=2023-01"
        ),
        (
            [PartitionField(source_id=10, field_id=1001, transform=MonthTransform(), name="partition_field")],
            [PartitionFieldValue(source_id=10, value=date(2023, 1, 1))],
            Record(partition_field=((2023-1970) * 12)),
            "partition_field=2023-01"
        ),
        # Month Transform
        (
            [PartitionField(source_id=8, field_id=1001, transform=YearTransform(), name="partition_field")],
            [PartitionFieldValue(source_id=8, value=datetime(2023, 1, 1, 11, 55, 59, 999999))],
            Record(partition_field=(2023-1970)),
            "partition_field=2023"
        ),
        (
            [PartitionField(source_id=9, field_id=1001, transform=YearTransform(), name="partition_field")],
            [PartitionFieldValue(source_id=9, value=datetime(2023, 1, 1, 11, 55, 59,999999, tzinfo=pytz.timezone('America/New_York')))],
            Record(partition_field=(2023-1970)),
            "partition_field=2023"
        ),
        (
            [PartitionField(source_id=10, field_id=1001, transform=YearTransform(), name="partition_field")],
            [PartitionFieldValue(source_id=10, value=date(2023, 1, 1))],
            Record(partition_field=(2023-1970)),
            "partition_field=2023"
        ),
        # Day Transform
        (
            [PartitionField(source_id=8, field_id=1001, transform=DayTransform(), name="partition_field")],
            [PartitionFieldValue(source_id=8, value=datetime(2023, 1, 1, 11, 55, 59, 999999))],
            Record(partition_field=19358),
            "partition_field=2023-01-01"
        ),
        (
            [PartitionField(source_id=9, field_id=1001, transform=DayTransform(), name="partition_field")],
            [PartitionFieldValue(source_id=9, value=datetime(2023, 1, 1, 11, 55, 59,999999, tzinfo=pytz.timezone('America/New_York')))],
            Record(partition_field=19358),
            "partition_field=2023-01-01"
        ),
        (
            [PartitionField(source_id=10, field_id=1001, transform=DayTransform(), name="partition_field")],
            [PartitionFieldValue(source_id=10, value=date(2023, 1, 1))],
            Record(partition_field=19358),
            "partition_field=2023-01-01"
        ),
        # Hour Transform
        (
            [PartitionField(source_id=8, field_id=1001, transform=HourTransform(), name="partition_field")],
            [PartitionFieldValue(source_id=8, value=datetime(2023, 1, 1, 11, 55, 59, 999999))],
            Record(partition_field=464603),
            "partition_field=2023-01-01-11"
        ),
        (
            [PartitionField(source_id=9, field_id=1001, transform=HourTransform(), name="partition_field")],
            [PartitionFieldValue(source_id=9, value=datetime(2023, 1, 1, 11, 55, 59,999999, tzinfo=pytz.timezone('America/New_York')))],
            Record(partition_field=464608), # 464608 = 464603 + 5, new york winter day light saving time
            "partition_field=2023-01-01-16"
        ),
    ]
)
def test_partition_key(arrow_table_with_null, partition_fields, partition_field_values, expected_partition_record, expected_hive_partition_path_slice) -> None:

    spec = PartitionSpec(*partition_fields)

    key = PartitionKey(
        raw_partition_field_values=partition_field_values,
        partition_spec=spec,
        schema=TABLE_SCHEMA,
    )
    # print(f"{key.partition=}")
    # print(f"{key.to_path()=}")
    # this affects the metadata in DataFile and all above layers
    assert key.partition == expected_partition_record 
    # this affects the parquet file path
    assert key.to_path() == expected_hive_partition_path_slice

