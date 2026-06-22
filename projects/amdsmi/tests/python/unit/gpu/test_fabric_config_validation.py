"""Fabric config mask-key validation unit tests (hardware-free)."""

import ctypes
import unittest

from common.common import amdsmi

_iface = amdsmi.amdsmi_interface
_wrap = amdsmi.amdsmi_wrapper


class TestFabricMaskKeyValidation(unittest.TestCase):
    """Hardware-free tests for _validate_fabric_mask_keys.

    The guard exists to reject a request whose mask selects a field the caller
    never supplied, which would otherwise write a zero-initialized value.
    """

    def test_masked_field_absent_raises(self):
        with self.assertRaises(_iface.AmdSmiParameterException):
            _iface._validate_fabric_mask_keys(
                _wrap.AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID, {}, _iface._FABRIC_PPOD_MASK_KEYS
            )

    def test_local_accels_requires_both_keys(self):
        bit = _wrap.AMDSMI_FABRIC_PPOD_FIELD_LOCAL_ACCELS
        # count present, list missing
        with self.assertRaises(_iface.AmdSmiParameterException):
            _iface._validate_fabric_mask_keys(
                bit, {"local_accelerator_count": 2}, _iface._FABRIC_PPOD_MASK_KEYS
            )
        # list present, count missing
        with self.assertRaises(_iface.AmdSmiParameterException):
            _iface._validate_fabric_mask_keys(
                bit, {"local_accelerators": [0, 1]}, _iface._FABRIC_PPOD_MASK_KEYS
            )
        # both present -> no raise
        _iface._validate_fabric_mask_keys(
            bit,
            {"local_accelerators": [0, 1], "local_accelerator_count": 2},
            _iface._FABRIC_PPOD_MASK_KEYS,
        )

    def test_all_masked_keys_present_ok(self):
        vpod_bits = (
            _wrap.AMDSMI_FABRIC_VPOD_FIELD_VPOD_ID | _wrap.AMDSMI_FABRIC_VPOD_FIELD_ADDR_MODE
        )
        _iface._validate_fabric_mask_keys(
            vpod_bits, {"vpod_id": 3, "addr_mode": 0}, _iface._FABRIC_VPOD_MASK_KEYS
        )

    def test_empty_mask_requires_nothing(self):
        for mask_keys in (
            _iface._FABRIC_PPOD_MASK_KEYS,
            _iface._FABRIC_VPOD_MASK_KEYS,
            _iface._FABRIC_STATION_MASK_KEYS,
        ):
            _iface._validate_fabric_mask_keys(0, {}, mask_keys)

    def test_undefined_mask_bit_raises(self):
        valid = 0
        for bit in _iface._FABRIC_PPOD_MASK_KEYS:
            valid |= bit
        undefined_bit = 1
        while undefined_bit & valid:
            undefined_bit <<= 1
        # A lone undefined bit is rejected.
        with self.assertRaises(_iface.AmdSmiParameterException):
            _iface._validate_fabric_mask_keys(undefined_bit, {}, _iface._FABRIC_PPOD_MASK_KEYS)
        # An undefined bit alongside a satisfied valid bit still raises.
        accel = _wrap.AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID
        with self.assertRaises(_iface.AmdSmiParameterException):
            _iface._validate_fabric_mask_keys(
                accel | undefined_bit, {"accelerator_id": 1}, _iface._FABRIC_PPOD_MASK_KEYS
            )

    def test_station_num_stations_gate(self):
        bit = _wrap.AMDSMI_FABRIC_DF_FIELD_NUM_STATIONS
        with self.assertRaises(_iface.AmdSmiParameterException):
            _iface._validate_fabric_mask_keys(bit, {}, _iface._FABRIC_STATION_MASK_KEYS)
        _iface._validate_fabric_mask_keys(
            bit, {"num_stations": 4}, _iface._FABRIC_STATION_MASK_KEYS
        )


class TestFabricPopulate(unittest.TestCase):
    """Hardware-free tests for _populate_fabric_config_data.

    The helper copies caller values into a ctypes payload; it must reject
    unknown fields and sequences that overflow a fixed-size array, and copy
    valid values element-wise.
    """

    @staticmethod
    def _ppod_struct():
        return _wrap.struct_amdsmi_fabric_ppod_data_t()

    def test_unknown_field_raises(self):
        with self.assertRaises(_iface.AmdSmiParameterException):
            _iface._populate_fabric_config_data(self._ppod_struct(), {"nonexistent": 1})

    def test_non_sequence_for_array_raises(self):
        with self.assertRaises(_iface.AmdSmiParameterException):
            _iface._populate_fabric_config_data(self._ppod_struct(), {"ppod_id": 5})

    def test_str_for_array_raises(self):
        # A str satisfies __len__ but the contract forbids it; without an explicit
        # str guard it would be iterated char-by-char into a confusing ctypes error.
        with self.assertRaises(_iface.AmdSmiParameterException):
            _iface._populate_fabric_config_data(self._ppod_struct(), {"local_accelerators": "012"})

    def test_bytes_for_array_ok(self):
        # bytes is an explicitly documented valid sequence and must still pass.
        struct = self._ppod_struct()
        _iface._populate_fabric_config_data(struct, {"local_accelerators": bytes([3, 4, 5])})
        self.assertEqual(list(struct.local_accelerators)[:3], [3, 4, 5])

    def test_oversized_sequence_raises(self):
        # local_accelerators is a 16-element array; 17 elements must be rejected.
        with self.assertRaises(_iface.AmdSmiParameterException):
            _iface._populate_fabric_config_data(
                self._ppod_struct(), {"local_accelerators": list(range(17))}
            )

    def test_valid_scalar_and_array_land_in_struct(self):
        struct = self._ppod_struct()
        _iface._populate_fabric_config_data(
            struct, {"accelerator_id": 7, "local_accelerators": [3, 4, 5]}
        )
        self.assertEqual(struct.accelerator_id, 7)
        self.assertEqual(list(struct.local_accelerators)[:3], [3, 4, 5])

    def test_array_element_out_of_range_raises(self):
        # ppod_id is c_ubyte*16; 256 must be rejected, not truncated to 0.
        with self.assertRaises(_iface.AmdSmiParameterException):
            _iface._populate_fabric_config_data(self._ppod_struct(), {"ppod_id": [256]})

    def test_scalar_out_of_range_raises(self):
        # accelerator_id is c_uint32.
        with self.assertRaises(_iface.AmdSmiParameterException):
            _iface._populate_fabric_config_data(self._ppod_struct(), {"accelerator_id": 2**32})

    def test_negative_into_unsigned_raises(self):
        # Docstring's truncation guard: -1 must not wrap into a large unsigned.
        with self.assertRaises(_iface.AmdSmiParameterException):
            _iface._populate_fabric_config_data(self._ppod_struct(), {"accelerator_id": -1})

    def test_max_boundary_values_accepted(self):
        struct = self._ppod_struct()
        _iface._populate_fabric_config_data(struct, {"accelerator_id": 2**32 - 1, "ppod_id": [255]})
        self.assertEqual(struct.accelerator_id, 2**32 - 1)
        self.assertEqual(struct.ppod_id[0], 255)

    def test_full_length_sequence_accepted(self):
        # Exactly the array length (16) is legal; only >16 overflows.
        struct = self._ppod_struct()
        _iface._populate_fabric_config_data(struct, {"local_accelerators": list(range(16))})
        self.assertEqual(list(struct.local_accelerators), list(range(16)))

    def test_check_int_range_signed_bounds(self):
        # No fabric struct field is signed, so exercise the signed branch directly.
        self.assertIsNone(_iface._check_fabric_int_range("k", 127, ctypes.c_int8))
        self.assertIsNone(_iface._check_fabric_int_range("k", -128, ctypes.c_int8))
        for bad in (128, -129):
            with self.assertRaises(_iface.AmdSmiParameterException):
                _iface._check_fabric_int_range("k", bad, ctypes.c_int8)


class TestFabricPpodPublicValidation(unittest.TestCase):
    """Validation in amdsmi_set_gpu_fabric_ppod_config that runs before the C call.

    A default-constructed handle is sufficient: these inputs are rejected during
    parameter validation, so control never reaches libamd_smi.
    """

    @staticmethod
    def _handle():
        return _wrap.amdsmi_processor_handle()

    def test_local_accelerator_count_exceeds_list_raises(self):
        with self.assertRaises(_iface.AmdSmiParameterException):
            _iface.amdsmi_set_gpu_fabric_ppod_config(
                self._handle(),
                _wrap.AMDSMI_FABRIC_PPOD_FIELD_LOCAL_ACCELS,
                {"local_accelerators": [12, 13], "local_accelerator_count": 4},
            )

    def test_non_bool_commit_raises(self):
        with self.assertRaises(_iface.AmdSmiParameterException):
            _iface.amdsmi_set_gpu_fabric_ppod_config(self._handle(), 0, {}, commit="false")


if __name__ == "__main__":
    unittest.main()
