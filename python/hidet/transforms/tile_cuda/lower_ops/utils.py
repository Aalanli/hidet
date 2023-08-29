from hidet.ir.dtypes import uint8, uint16, uint32, uint64
from hidet.ir.type import PointerType, DataType, sizeof


def get_type_erased_dtype(ptr_type: PointerType) -> DataType:
    # get the type-erased data type of the loaded element
    assert isinstance(ptr_type, PointerType)
    nbits: int = sizeof(ptr_type.base_type) * 8
    nbits2dtype = {
        8: uint8,
        16: uint16,
        32: uint32,
        64: uint64
    }
    return nbits2dtype[nbits]

