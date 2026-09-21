from __future__ import annotations

import argparse
from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "indoor_localization"))
from profile_config import apply_profile_defaults


class ProfileConfigTests(unittest.TestCase):
    def test_nested_yaml_applies_typed_defaults(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--keyframe-rate-hz", type=float, default=5.0)
        parser.add_argument(
            "--enable-scan-recovery",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.yaml"
            path.write_text(
                "timing:\n  keyframe_rate_hz: 3\n"
                "recovery:\n  enable_scan_recovery: true\n",
                encoding="utf-8",
            )
            apply_profile_defaults(parser, path)
        args = parser.parse_args([])
        self.assertEqual(args.keyframe_rate_hz, 3.0)
        self.assertTrue(args.enable_scan_recovery)

    def test_unknown_key_is_rejected(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--known", type=int, default=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.yaml"
            path.write_text("group:\n  typo: 2\n", encoding="utf-8")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                apply_profile_defaults(parser, path)


if __name__ == "__main__":
    unittest.main()
