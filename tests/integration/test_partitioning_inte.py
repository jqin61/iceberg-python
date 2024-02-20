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
# pylint:disable=redefined-outer-name
import os
import time
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import pytz
from pyarrow.fs import S3FileSystem
from pyspark.sql import SparkSession
from pytest_mock.plugin import MockerFixture

from pyiceberg.catalog import Catalog, Properties, Table, load_catalog
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.exceptions import NamespaceAlreadyExistsError, NoSuchTableError
from pyiceberg.schema import Schema
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


@pytest.fixture()
def catalog() -> Catalog:
    catalog = load_catalog(
        "local",
        **{
            "type": "rest",
            "uri": "http://localhost:8181",
            "s3.endpoint": "http://localhost:9000",
            "s3.access-key-id": "admin",
            "s3.secret-access-key": "password",
        },
    )

    try:
        catalog.create_namespace("default")
    except NamespaceAlreadyExistsError:
        pass

    return catalog


@pytest.fixture(scope="session")
def session_catalog() -> Catalog:
    return load_catalog(
        "local",
        **{
            "type": "rest",
            "uri": "http://localhost:8181",
            "s3.endpoint": "http://localhost:9000",
            "s3.access-key-id": "admin",
            "s3.secret-access-key": "password",
        },
    )


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    import importlib.metadata
    import os

    spark_version = ".".join(importlib.metadata.version("pyspark").split(".")[:2])
    scala_version = "2.12"
    iceberg_version = "1.4.3"

    os.environ["PYSPARK_SUBMIT_ARGS"] = (
        f"--packages org.apache.iceberg:iceberg-spark-runtime-{spark_version}_{scala_version}:{iceberg_version},"
        f"org.apache.iceberg:iceberg-aws-bundle:{iceberg_version} pyspark-shell"
    )
    os.environ["AWS_REGION"] = "us-east-1"
    os.environ["AWS_ACCESS_KEY_ID"] = "admin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "password"

    spark = (
        SparkSession.builder.appName("PyIceberg integration test")
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config("spark.sql.catalog.integration", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.integration.catalog-impl", "org.apache.iceberg.rest.RESTCatalog")
        .config("spark.sql.catalog.integration.uri", "http://localhost:8181")
        .config("spark.sql.catalog.integration.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
        .config("spark.sql.catalog.integration.warehouse", "s3://warehouse/wh/")
        .config("spark.sql.catalog.integration.s3.endpoint", "http://localhost:9000")
        .config("spark.sql.catalog.integration.s3.path-style-access", "true")
        .config("spark.sql.defaultCatalog", "integration")
        .getOrCreate()
    )

    return spark


@pytest.mark.london
def test_load_table_and_spark_query(session_catalog: Catalog, spark:SparkSession) -> Table:
    identifier = "default.test_table"
    try:
        session_catalog.drop_table(identifier=identifier)
    except NoSuchTableError:
        pass

    create_sql = f"""CREATE TABLE {identifier} (
        id int,
        other_data string
    )
    USING iceberg
    PARTITIONED BY (
        truncate(id, 10)  -- Truncating 'id' integer column to a width of 10
    )
    """
    spark.sql(create_sql)

    iceberg_table = session_catalog.load_table(identifier=identifier)
    print(iceberg_table.current_snapshot())


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

identifier = "default.test_table"
@pytest.mark.parametrize(
    "partition_fields, partition_field_values, expected_partition_record, expected_hive_partition_path_slice, equivalent_spark_table_sql, equivalent_spark_data_sql", 
    [
        # Identity Transform
        (
            [PartitionField(source_id=1, field_id=1001, transform=IdentityTransform(), name="boolean_field")],
            [PartitionFieldValue(source_id=1, value=False)],
            Record(boolean_field=False),
            "boolean_field=False", #pyiceberg writes False while spark writes false, so verification failed
            None,
            None,
            # f"""CREATE TABLE {identifier} (
            #     boolean_field boolean,
            #     string_field string
            # )
            # USING iceberg
            # PARTITIONED BY (
            #     identity(boolean_field)  -- Partitioning by 'boolean_field'
            # )
            # """,
            # f"""INSERT INTO {identifier}
            # VALUES
            # (false, 'Boolean field set to false');
            # """
        ),
        (
            [PartitionField(source_id=2, field_id=1001, transform=IdentityTransform(), name="string_field")],
            [PartitionFieldValue(source_id=2, value="sample_string")],
            Record(string_field="sample_string"),
            "string_field=sample_string",
            f"""CREATE TABLE {identifier} (
                string_field string,
                another_string_field string
            )
            USING iceberg
            PARTITIONED BY (
                identity(string_field)
            )
            """,
            f"""INSERT INTO {identifier}
            VALUES
            ('sample_string', 'Another string value')
            """
        ),
        (
            [PartitionField(source_id=4, field_id=1001, transform=IdentityTransform(), name="int_field")],
            [PartitionFieldValue(source_id=4, value=42)],
            Record(int_field=42),
            "int_field=42",
            f"""CREATE TABLE {identifier} (
                int_field int,
                string_field string
            )
            USING iceberg
            PARTITIONED BY (
                identity(int_field)
            )
            """,
            f"""INSERT INTO {identifier}
            VALUES
            (42, 'Associated string value for int 42')
            """
        ),
        (
            [PartitionField(source_id=5, field_id=1001, transform=IdentityTransform(), name="long_field")],
            [PartitionFieldValue(source_id=5, value=1234567890123456789)],
            Record(long_field=1234567890123456789),
            "long_field=1234567890123456789",
            f"""CREATE TABLE {identifier} (
                long_field bigint,
                string_field string
            )
            USING iceberg
            PARTITIONED BY (
                identity(long_field)
            )
            """,
            f"""INSERT INTO {identifier}
            VALUES
            (1234567890123456789, 'Associated string value for long 1234567890123456789')
            """
        ),
        (
            [PartitionField(source_id=6, field_id=1001, transform=IdentityTransform(), name="float_field")],
            [PartitionFieldValue(source_id=6, value=3.14)],
            Record(float_field=3.14),
            "float_field=3.14",
            #spark writes differently as pyiceberg, Record[float_field=3.140000104904175] path:float_field=3.14 (Record has difference)
            None,None, 
            # f"""CREATE TABLE {identifier} (
            #     float_field float,
            #     string_field string
            # )
            # USING iceberg
            # PARTITIONED BY (
            #     identity(float_field)
            # )
            # """,
            # f"""INSERT INTO {identifier}
            # VALUES
            # (3.14, 'Associated string value for float 3.14')
            # """
        ),
        (
            [PartitionField(source_id=7, field_id=1001, transform=IdentityTransform(), name="double_field")],
            [PartitionFieldValue(source_id=7, value=6.282)],
            Record(double_field=6.282), 
            "double_field=6.282",
            #spark writes differently as pyiceberg, Record[double_field=6.2820000648498535] path:double_field=6.282 (Record has difference)
            None, None
            # f"""CREATE TABLE {identifier} (
            #     double_field double,
            #     string_field string
            # )
            # USING iceberg
            # PARTITIONED BY (
            #     identity(double_field)
            # )
            # """,
            # f"""INSERT INTO {identifier}
            # VALUES
            # (6.282, 'Associated string value for double 6.282')
            # """
        ),
        (
            [PartitionField(source_id=8, field_id=1001, transform=IdentityTransform(), name="timestamp_field")],
            [PartitionFieldValue(source_id=8, value=datetime(2023, 1, 1, 12, 0, 0))],
            Record(timestamp_field=1672574400000000),
            "timestamp_field=2023-01-01T12%3A00%3A00",
            #spark writes differently as pyiceberg, Record[timestamp_field=1672574400000000] path:timestamp_field=2023-01-01T12%3A00Z  (the Z is the difference)
            None,  
            None
            # f"""CREATE TABLE {identifier} (
            #     timestamp_field timestamp,
            #     string_field string
            # )
            # USING iceberg
            # PARTITIONED BY (
            #     identity(timestamp_field)
            # )
            # """,
            # f"""INSERT INTO {identifier}
            # VALUES
            # (CAST('2023-01-01 12:00:00' AS TIMESTAMP), 'Associated string value for timestamp 2023-01-01T12:00:00')
            # """
        ),
        (
            [PartitionField(source_id=10, field_id=1001, transform=IdentityTransform(), name="date_field")],
            [PartitionFieldValue(source_id=10, value=date(2023, 1, 1))],
            Record(date_field=19358),
            "date_field=2023-01-01",
            f"""CREATE TABLE {identifier} (
                date_field date,
                string_field string
            )
            USING iceberg
            PARTITIONED BY (
                identity(date_field)
            )
            """,
            f"""INSERT INTO {identifier}
            VALUES
            (CAST('2023-01-01' AS DATE), 'Associated string value for date 2023-01-01')
            """,
        ),
        (
            [PartitionField(source_id=11, field_id=1001, transform=IdentityTransform(), name="binary_field")],
            [PartitionFieldValue(source_id=11, value=b'example')],
            Record(binary_field=b'example'),
            "binary_field=ZXhhbXBsZQ%3D%3D",  
            f"""CREATE TABLE {identifier} (
                binary_field binary,
                string_field string
            )
            USING iceberg
            PARTITIONED BY (
                identity(binary_field)
            )
            """,
            f"""INSERT INTO {identifier}
            VALUES
            (CAST('example' AS BINARY), 'Associated string value for binary `example`')
            """
        ),
        # Year Month Day Hour Transform
        # Month Transform
        (
            [PartitionField(source_id=8, field_id=1001, transform=MonthTransform(), name="event_timestamp_month")],
            [PartitionFieldValue(source_id=8, value=datetime(2023, 1, 1, 11, 55, 59, 999999))],
            Record(event_timestamp_month=((2023-1970) * 12)),
            "event_timestamp_month=2023-01",
            f"""CREATE TABLE {identifier} (
                event_timestamp timestamp,
                other_data string
            )
            USING iceberg
            PARTITIONED BY (
                month(event_timestamp)  -- Partitioning by month from 'event_timestamp'
            )
            """,
            f"""INSERT INTO {identifier}
            VALUES
            (CAST('2023-01-01 11:55:59.999999' AS TIMESTAMP), 'Event at 2023-01-01 11:55:59.999999');
            """
        ),
        (
            [PartitionField(source_id=9, field_id=1001, transform=MonthTransform(), name="partition_field")],
            [PartitionFieldValue(source_id=9, value=datetime(2023, 1, 1, 11, 55, 59,999999, tzinfo=pytz.timezone('America/New_York')))],
            Record(partition_field=((2023-1970) * 12)),
            "partition_field=2023-01",
            None,
            None
        ),
        (
            [PartitionField(source_id=10, field_id=1001, transform=MonthTransform(), name="event_date_month")],
            [PartitionFieldValue(source_id=10, value=date(2023, 1, 1))],
            Record(event_date_month=((2023-1970) * 12)),
            "event_date_month=2023-01",
            f"""CREATE TABLE {identifier} (
                event_date date,
                other_data string
            )
            USING iceberg
            PARTITIONED BY (
                month(event_date)  -- Partitioning by month from 'event_date'
            )
            """,
            f"""INSERT INTO {identifier}
            VALUES
            (CAST('2023-01-01' AS DATE), 'Event on 2023-01-01');
            """
        ),
        # Year Transform
        (
            [PartitionField(source_id=8, field_id=1001, transform=YearTransform(), name="event_timestamp_year")],
            [PartitionFieldValue(source_id=8, value=datetime(2023, 1, 1, 11, 55, 59, 999999))],
            Record(event_timestamp_year=(2023-1970)),
            "event_timestamp_year=2023",
            f"""CREATE TABLE {identifier}  (
                event_timestamp timestamp,
                other_data string
            )
            USING iceberg
            PARTITIONED BY (
                year(event_timestamp)  -- Partitioning by year from 'event_timestamp'
            )
            """,
            f"""INSERT INTO {identifier} 
            VALUES
            (CAST('2023-01-01 11:55:59.999999' AS TIMESTAMP), 'Event at 2023-01-01 11:55:59.999999');
            """

        ),
        (
            [PartitionField(source_id=9, field_id=1001, transform=YearTransform(), name="partition_field")],
            [PartitionFieldValue(source_id=9, value=datetime(2023, 1, 1, 11, 55, 59,999999, tzinfo=pytz.timezone('America/New_York')))],
            Record(partition_field=(2023-1970)),
            "partition_field=2023",
            None,
            None,
        ),
        (
            [PartitionField(source_id=10, field_id=1001, transform=YearTransform(), name="event_date_year")],
            [PartitionFieldValue(source_id=10, value=date(2023, 1, 1))],
            Record(event_date_year=(2023-1970)),
            "event_date_year=2023",
            f"""CREATE TABLE {identifier} (
                event_date date,
                other_data string
            )
            USING iceberg
            PARTITIONED BY (
                year(event_date)  -- Partitioning by year from 'event_date'
            )
            """,
            f"""INSERT INTO {identifier} 
            VALUES
            (CAST('2023-01-01' AS DATE), 'Event on 2023-01-01');
            """
        ),
        # Day Transform
        (
            [PartitionField(source_id=8, field_id=1001, transform=DayTransform(), name="event_timestamp_day")],
            [PartitionFieldValue(source_id=8, value=datetime(2023, 1, 1, 11, 55, 59, 999999))],
            Record(event_timestamp_day=19358),
            "event_timestamp_day=2023-01-01",
            f"""CREATE TABLE {identifier} (
                event_timestamp timestamp,
                other_data string
            )
            USING iceberg
            PARTITIONED BY (
                day(event_timestamp)  -- Partitioning by day from 'event_timestamp'
            )
            """,
            f"""INSERT INTO {identifier}
            VALUES
            (CAST('2023-01-01' AS DATE), 'Event on 2023-01-01');
            """
        
        ),
        (
            [PartitionField(source_id=9, field_id=1001, transform=DayTransform(), name="partition_field")],
            [PartitionFieldValue(source_id=9, value=datetime(2023, 1, 1, 11, 55, 59,999999, tzinfo=pytz.timezone('America/New_York')))],
            Record(partition_field=19358),
            "partition_field=2023-01-01",
            None,
            None
        ),
        (
            [PartitionField(source_id=10, field_id=1001, transform=DayTransform(), name="event_date_day")],
            [PartitionFieldValue(source_id=10, value=date(2023, 1, 1))],
            Record(event_date_day=19358),
            "event_date_day=2023-01-01",
            f"""CREATE TABLE {identifier} (
                event_date date,
                other_data string
            )
            USING iceberg
            PARTITIONED BY (
                day(event_date)  -- Partitioning by day from 'event_date'
            )
            """,
            f"""INSERT INTO {identifier}
            VALUES
            (CAST('2023-01-01' AS DATE), 'Event on 2023-01-01');
            """,
        ),
        # Hour Transform
        (
            [PartitionField(source_id=8, field_id=1001, transform=HourTransform(), name="event_timestamp_hour")],
            [PartitionFieldValue(source_id=8, value=datetime(2023, 1, 1, 11, 55, 59, 999999))],
            Record(event_timestamp_hour=464603),
            "event_timestamp_hour=2023-01-01-11",
            f"""CREATE TABLE {identifier} (
                event_timestamp timestamp,
                other_data string
            )
            USING iceberg
            PARTITIONED BY (
                hour(event_timestamp)  -- Partitioning by hour from 'event_timestamp'
            )
            """,
            f"""INSERT INTO {identifier}
                VALUES
                (CAST('2023-01-01 11:55:59.999999' AS TIMESTAMP), 'Event within the 11th hour of 2023-01-01');
                """
        ),
        (
            [PartitionField(source_id=9, field_id=1001, transform=HourTransform(), name="partition_field")],
            [PartitionFieldValue(source_id=9, value=datetime(2023, 1, 1, 11, 55, 59,999999, tzinfo=pytz.timezone('America/New_York')))],
            Record(partition_field=464608), # 464608 = 464603 + 5, new york winter day light saving time
            "partition_field=2023-01-01-16",
            None,
            None
        ),
        # Truncate Transform
        (
            [PartitionField(source_id=4, field_id=1001, transform=TruncateTransform(10), name="id_trunc")],
            [PartitionFieldValue(source_id=4, value=12345)],
            Record(id_trunc=12340),
            "id_trunc=12340",
            f"""CREATE TABLE {identifier} (
                id int,
                other_data string
            )
            USING iceberg
            PARTITIONED BY (
                truncate(id, 10)  -- Truncating 'id' integer column to a width of 10
            )
            """,
            f"""INSERT INTO {identifier}
                VALUES
                (12345, 'Sample data for int');
            """

        ),
        (
            [PartitionField(source_id=5, field_id=1001, transform=TruncateTransform(2), name="bigint_field_trunc")],
            [PartitionFieldValue(source_id=5, value=2**32 + 1)],
            Record(bigint_field_trunc=2**32), #4294967296
            "bigint_field_trunc=4294967296",
            f"""CREATE TABLE {identifier} (
                bigint_field bigint,
                other_data string
            )
            USING iceberg
            PARTITIONED BY (
                truncate(bigint_field, 2)  -- Truncating 'bigint_field' long column to a width of 2
            )
            """,
            f"""INSERT INTO {identifier}
            VALUES
            (4294967297, 'Sample data for long');
            """
        ),
        (
            [PartitionField(source_id=2, field_id=1001, transform=TruncateTransform(3), name="string_field_trunc")],
            [PartitionFieldValue(source_id=2, value="abcdefg")],
            Record(string_field_trunc="abc"),
            "string_field_trunc=abc",
            f"""CREATE TABLE {identifier} (
                string_field string,
                other_data string
            )
            USING iceberg
            PARTITIONED BY (
                truncate(string_field, 3)  -- Truncating 'string_field' string column to a length of 3 characters
            )
            """,
            f"""INSERT INTO {identifier}
            VALUES
            ('abcdefg', 'Another sample for string');
            """
        ),
        # it seems the transform.tohumanstring does take a bytes type which means i do not need to do extra conversion in iceberg_typed_value() for BinaryType
        (
            [PartitionField(source_id=11, field_id=1001, transform=TruncateTransform(10), name="binary_field_trunc")],
            [PartitionFieldValue(source_id=11, value=b'HELLOICEBERG')],
            Record(binary_field_trunc=b'HELLOICEBE'),
            "binary_field_trunc=SEVMTE9JQ0VCRQ%3D%3D",
            f"""CREATE TABLE {identifier} (
                binary_field binary,
                other_data string
            )
            USING iceberg
            PARTITIONED BY (
                truncate(binary_field, 10)  -- Truncating 'binary_field' binary column to a length of 10 bytes
            )
            """,
            f"""INSERT INTO {identifier}
                VALUES
                (binary('HELLOICEBERG'), 'Sample data for binary');
            """
        ),
        # Bucket Transform
        (
            [PartitionField(source_id=4, field_id=1001, transform=BucketTransform(2), name="int_field_bucket")],
            [PartitionFieldValue(source_id=4, value=10)],
            Record(int_field_bucket=0),
            "int_field_bucket=0",
            f"""CREATE TABLE {identifier} (
                int_field int,
                other_data string
            )
            USING iceberg
            PARTITIONED BY (
                bucket(2, int_field)  -- Distributing 'int_field' across 2 buckets
            )
            """,
            f"""INSERT INTO {identifier}
            VALUES
            (10, 'Integer with value 10');
            """,
        ),
        # Test multiple field combinations could generate the Partition record and hive partition path correctly
        (
            [
                PartitionField(source_id=8, field_id=1001, transform=YearTransform(), name="timestamp_field_year"),
                PartitionField(source_id=10, field_id=1002, transform=DayTransform(), name="date_field_day")
            ],
            [
                PartitionFieldValue(source_id=8, value=datetime(2023, 1, 1, 11, 55, 59, 999999)),
                PartitionFieldValue(source_id=10, value=date(2023, 1, 1)),
            ],
            Record(timestamp_field_year=53, date_field_day=19358),
            "timestamp_field_year=2023/date_field_day=2023-01-01",
            f"""CREATE TABLE {identifier} (
                timestamp_field timestamp,
                date_field date,
                string_field string
            )
            USING iceberg
            PARTITIONED BY (
                year(timestamp_field),
                day(date_field)
            )
            """,
            f"""INSERT INTO {identifier}
            VALUES
            (CAST('2023-01-01 11:55:59.999999' AS TIMESTAMP), CAST('2023-01-01' AS DATE), 'some data');
            """,
        ),
    ]
)
@pytest.mark.greate
def test_partition_key(session_catalog: Catalog, spark:SparkSession, arrow_table_with_null: pa.table, partition_fields, partition_field_values, expected_partition_record, expected_hive_partition_path_slice, equivalent_spark_table_sql, equivalent_spark_data_sql) -> None:

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
    # this affects the hive partitioning part in the parquet file path
    assert key.to_path() == expected_hive_partition_path_slice

    from pyspark.sql.utils import AnalysisException

    #### verify expected values are not made up but conform to spark behaviors
    if equivalent_spark_table_sql is not None and equivalent_spark_data_sql is not None:
        try:
            spark.sql(f"drop table {identifier}")
        except AnalysisException:
            pass
        
        spark.sql(equivalent_spark_table_sql)
        spark.sql(equivalent_spark_data_sql)

        iceberg_table = session_catalog.load_table(identifier=identifier)
        verify_partition = iceberg_table.current_snapshot().manifests(iceberg_table.io)[0].fetch_manifest_entry(iceberg_table.io)[0].data_file.partition
        verify_path = iceberg_table.current_snapshot().manifests(iceberg_table.io)[0].fetch_manifest_entry(iceberg_table.io)[0].data_file.file_path
        # print(f"{verify_partition=}")
        # print(f"{verify_path=}")
        assert verify_partition == expected_partition_record
        assert expected_hive_partition_path_slice in verify_path