import unittest

from adaptive_vision_rl.protocol import extract_answer_candidate, parse_action


class ProtocolTests(unittest.TestCase):
    def test_direct_answer_and_boxed_value(self):
        action = parse_action(
            r"<think>I can read it.</think><answer>\boxed{659}</answer>",
            allow_tool=True,
            image_size=(400, 300),
        )
        self.assertTrue(action.valid)
        self.assertEqual(action.kind, "answer")
        self.assertEqual(action.answer, "659")

    def test_valid_tool_call(self):
        action = parse_action(
            '<think>I need detail.</think><tool_call>{"name":"request_local_region",'
            '"arguments":{"bbox_2d":[10,20,200,250]}}</tool_call>',
            allow_tool=True,
            image_size=(400, 300),
        )
        self.assertTrue(action.valid)
        self.assertEqual(action.kind, "tool")
        self.assertEqual(action.bbox, (10.0, 20.0, 200.0, 250.0))

    def test_rejects_second_or_out_of_bounds_tool(self):
        text = (
            '<think>crop</think><tool_call>{"name":"request_local_region",'
            '"arguments":{"bbox_2d":[0,0,401,100]}}</tool_call>'
        )
        self.assertFalse(parse_action(text, allow_tool=True, image_size=(400, 300)).valid)
        valid = text.replace("401", "400")
        self.assertFalse(parse_action(valid, allow_tool=False, image_size=(400, 300)).valid)

    def test_rejects_extra_text_and_bad_schema(self):
        self.assertFalse(
            parse_action(
                "prefix <think>x</think><answer>1</answer>",
                allow_tool=True,
                image_size=(10, 10),
            ).valid
        )

    def test_answer_accuracy_can_be_decoupled_from_format(self):
        malformed = r"<answer>\boxed{42}</answer>"
        self.assertFalse(
            parse_action(malformed, allow_tool=True, image_size=(10, 10)).valid
        )
        self.assertEqual(extract_answer_candidate(malformed), "42")
        self.assertFalse(
            parse_action(
                '<think>x</think><tool_call>{"name":"other","arguments":{}}</tool_call>',
                allow_tool=True,
                image_size=(10, 10),
            ).valid
        )


if __name__ == "__main__":
    unittest.main()
