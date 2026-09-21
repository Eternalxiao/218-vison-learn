import struct
import unittest

from coordinate_validator import Mapping, validate_points
from gcode_demo import frame_block, generate_move, split_blocks
from protocol import BinaryFrameParser, decode_json, decode_text, encode_binary, encode_json, encode_text


class ProtocolTests(unittest.TestCase):
    def test_text_and_json_round_trip(self):
        expected = {"class_id": 1, "u": 320, "v": 224, "angle": 15.5}
        self.assertEqual(decode_text(encode_text(1, 320, 224, 15.5)), expected)
        self.assertEqual(decode_json(encode_json(1, 320, 224, 15.5)), expected)

    def test_binary_partial_and_concatenated_frames(self):
        first = encode_binary(1, struct.pack(">hh", 320, 224))
        second = encode_binary(2, b"done")
        parser = BinaryFrameParser()
        self.assertEqual(parser.feed(first[:3]), [])
        self.assertEqual(parser.feed(first[3:] + second), [(1, struct.pack(">hh", 320, 224)), (2, b"done")])

    def test_binary_parser_recovers_after_bad_checksum(self):
        damaged = bytearray(encode_binary(1, b"bad"))
        damaged[-3] ^= 0x01
        valid = encode_binary(2, b"good")
        self.assertEqual(BinaryFrameParser().feed(damaged + valid), [(2, b"good")])

    def test_little_endian_frame(self):
        frame = encode_binary(3, b"little", endian="little")
        self.assertEqual(BinaryFrameParser(endian="little").feed(frame), [(3, b"little")])


class CoordinateTests(unittest.TestCase):
    def setUp(self):
        self.mapping = Mapping(2.0, 320.0, 224.0, 271.0, 221.0, -1, 1)

    def test_center_and_axis_swap(self):
        self.assertEqual(self.mapping.pixel_to_machine(320, 224), (271.0, 221.0))
        self.assertEqual(self.mapping.pixel_to_machine(340, 244), (261.0, 231.0))

    def test_residual_summary(self):
        results, rmse, maximum = validate_points(self.mapping, [(340, 244, 261, 231)])
        self.assertEqual(results[0]["error"], 0.0)
        self.assertEqual(rmse, 0.0)
        self.assertEqual(maximum, 0.0)

    def test_mapping_rejects_invalid_scale_and_sign(self):
        with self.assertRaises(ValueError):
            Mapping(0, 320, 224, 271, 221, -1, 1)
        with self.assertRaises(ValueError):
            Mapping(2, 320, 224, 271, 221, 0, 1)

    def test_gcode_is_split_after_sync_commands(self):
        blocks = split_blocks(generate_move(self.mapping, 340, 244, 3000))
        self.assertEqual(blocks, [["G90", "G1 X261.000 Y231.000 F3000", "M400"], ["G4 P500"]])
        framed = frame_block(blocks[0])
        self.assertTrue(framed.startswith("EEEF\r\n"))
        self.assertTrue(framed.endswith("FFFE\r\n"))


if __name__ == "__main__":
    unittest.main()
