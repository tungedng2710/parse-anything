import unittest

from reward import (
    content_reward,
    exact_reward,
    format_reward,
    parse_table_html,
    reasoning_metadata_reward,
    structure_reward,
)


REFERENCE = """<table><thead><tr><th colspan="2">Name</th></tr></thead>
<tbody><tr><td>A</td><td>B</td></tr></tbody></table>"""


class RewardTest(unittest.TestCase):
    def test_parser_recovers_shape_and_spans(self):
        parsed = parse_table_html(REFERENCE)
        self.assertTrue(parsed.valid)
        self.assertEqual((parsed.num_rows, parsed.num_cols, parsed.num_cells), (2, 2, 3))
        self.assertTrue(parsed.has_merged_cells)

    def test_trailing_rowspan_extends_logical_row_count(self):
        parsed = parse_table_html(
            '<table><tr><td rowspan="3">A</td><td>B</td></tr></table>'
        )
        self.assertEqual((parsed.num_rows, parsed.num_cols), (3, 2))

    def test_format_rejects_extra_text_and_incomplete_html(self):
        rewards = format_reward([REFERENCE, f"answer: {REFERENCE}", "<table><tr><td>x"])
        self.assertEqual(rewards, [1.0, 0.0, 0.0])

    def test_invalid_wrappers_receive_no_dense_reward(self):
        for wrapped in (f"```html\n{REFERENCE}\n```", f"<!doctype html>{REFERENCE}"):
            self.assertEqual(format_reward([wrapped]), [0.0])
            self.assertEqual(structure_reward([wrapped], [REFERENCE]), [0.0])
            self.assertEqual(content_reward([wrapped], [REFERENCE]), [0.0])

    def test_exact_ignores_indentation_and_attribute_order(self):
        reference = '<table><tr><td rowspan="1" colspan="2">x</td></tr></table>'
        prediction = '<table>\n<tr><td colspan="2" rowspan="1"> x </td></tr>\n</table>'
        self.assertEqual(exact_reward([prediction], [reference]), [1.0])

    def test_reference_may_be_wrapped_in_an_html_document(self):
        wrapped = f"<html><head><style>table {{ color: black; }}</style></head><body>{REFERENCE}</body></html>"
        self.assertEqual(exact_reward([REFERENCE], [wrapped]), [1.0])
        self.assertEqual(format_reward([wrapped]), [0.0])

    def test_structure_and_content_rewards_are_separate(self):
        wrong_text = REFERENCE.replace(">A<", ">X<")
        wrong_span = REFERENCE.replace(' colspan="2"', "")
        self.assertEqual(structure_reward([wrong_text], [REFERENCE]), [1.0])
        self.assertLess(content_reward([wrong_text], [REFERENCE])[0], 1.0)
        self.assertLess(structure_reward([wrong_span], [REFERENCE])[0], 1.0)
        self.assertEqual(content_reward([wrong_span], [REFERENCE]), [1.0])

    def test_reasoning_metadata_reward_uses_trace_facts(self):
        score = reasoning_metadata_reward(
            [REFERENCE],
            num_rows=[2],
            num_cols=[2],
            num_cells=[3],
            has_merged_cells=[True],
        )
        self.assertEqual(score, [1.0])

    def test_conversational_completion(self):
        completion = [{"role": "assistant", "content": REFERENCE}]
        self.assertEqual(format_reward([completion]), [1.0])


if __name__ == "__main__":
    unittest.main()
