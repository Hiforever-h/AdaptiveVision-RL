import unittest
from types import SimpleNamespace

from adaptive_vision_rl.thinking_template import (
    ASSISTANT_PREFIX,
    apply_thinking_chat_template,
    configure_thinking_tokenizer,
)


class ThinkingTemplateTests(unittest.TestCase):
    def test_replaces_only_generation_prefill(self):
        template = (
            "{{- '<|im_start|>assistant\\n<think>\\n' + content }}\n"
            "{%- if add_generation_prompt %}\n"
            " {{- '<|im_start|>assistant\\n<think>\\n' }}\n"
            "{%- endif %}\n"
        )
        tokenizer = SimpleNamespace(chat_template=template)
        configure_thinking_tokenizer(tokenizer)
        self.assertIn("assistant\\n<think>\\n' + content", tokenizer.chat_template)
        self.assertTrue(
            tokenizer.chat_template.endswith(
                "{%- if add_generation_prompt %}\n"
                " {{- '<|im_start|>assistant\\n' }}\n"
                "{%- endif %}\n"
            )
        )

    def test_rejects_non_thinking_template(self):
        tokenizer = SimpleNamespace(
            chat_template="{%- if add_generation_prompt %}\n"
            " {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"
        )
        with self.assertRaisesRegex(ValueError, "Thinking generation prefill"):
            configure_thinking_tokenizer(tokenizer)

    def test_render_requires_bare_assistant_prefix(self):
        class FakeTokenizer:
            def __init__(self, suffix):
                self.suffix = suffix

            def apply_chat_template(self, messages, **kwargs):
                if kwargs != {"add_generation_prompt": True, "tokenize": False}:
                    raise AssertionError(kwargs)
                return "<|im_start|>user\nquestion<|im_end|>\n" + self.suffix

        messages = [{"role": "user", "content": "question"}]
        prompt = apply_thinking_chat_template(FakeTokenizer(ASSISTANT_PREFIX), messages)
        self.assertTrue(prompt.endswith(ASSISTANT_PREFIX))
        with self.assertRaisesRegex(ValueError, "without a prefilled <think>"):
            apply_thinking_chat_template(
                FakeTokenizer(ASSISTANT_PREFIX + "<think>\n"), messages
            )


if __name__ == "__main__":
    unittest.main()
