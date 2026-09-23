"""Shared prompts for training rollouts and standalone evaluation."""

INITIAL_PROMPT = """<image>
You are given a low-resolution version of an image and a question.

Question: {question}
Low-resolution image size: width={width}, height={height}.

Output exactly one action, with no text outside its tag:
1. Answer directly with <answer>...</answer>; or
2. Request one high-resolution crop with:
<tool_call>{{"name":"request_local_region","arguments":{{"bbox_2d":[x1,y1,x2,y2]}}}}</tool_call>

The bounding box uses xyxy coordinates normalized to the integer range 0 to 1000,
independent of the displayed image size. The origin is the top-left corner. The right
and bottom coordinates are exclusive. You may call the tool at most once. Do not output
an answer in the same turn as a tool call. You may optionally place one non-empty
reasoning block enclosed by <think> and </think> immediately before the action tag.
"""


SECOND_PROMPT = """<image>
This is the same low-resolution full image.

<image>
This is the requested high-resolution crop.

Question: {question}

Use both images. Output the final response as <answer>...</answer>, with no text outside
the tag. You may optionally place one non-empty reasoning block enclosed by <think>
and </think> immediately before the answer. You cannot call another tool.
"""
