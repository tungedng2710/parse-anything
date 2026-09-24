import unittest
from unittest.mock import patch

from sft import ConversationDataset, parse_args, validate_args


class SftArgumentsTest(unittest.TestCase):
    def test_defaults_use_chandra_and_sft_output(self):
        args = parse_args([])
        self.assertEqual(args.model_name, "datalab-to/chandra-ocr-2")
        self.assertEqual(args.output_dir, "chandra_ocr_2_table_html_sft")
        validate_args(args)

    def test_invalid_warmup_ratio_is_rejected(self):
        args = parse_args(["--warmup-ratio", "1"])
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_conversations_are_materialized_lazily(self):
        dataset = [{"row": 0}, {"row": 1}]
        conversations = ConversationDataset(dataset, "instruction")
        with patch("sft.to_conversation", return_value={"messages": []}) as convert:
            self.assertEqual(len(conversations), 2)
            convert.assert_not_called()
            self.assertEqual(conversations[1], {"messages": []})
            convert.assert_called_once_with({"row": 1}, "instruction")


if __name__ == "__main__":
    unittest.main()
