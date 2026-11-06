from feast import Entity
from feast.value_type import ValueType


customer = Entity(
    name="customer",
    join_keys = ["id"],
    value_type = ValueType.INT32,
    description = "The customer of the bank"
)