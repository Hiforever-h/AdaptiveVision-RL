"""Shared prompts for training rollouts and standalone evaluation."""

INITIAL_PROMPT = """<image>
You see a low-resolution full image and a question.

Question: {question}
Low-resolution image size: width={width}, height={height}.

Your response MUST start with <think> and contain a non-empty </think> block.
Briefly reason only from details visible in the low-resolution image.
After </think>, output exactly ONE action:

1. If the answer is reliably readable now, output <answer>short answer</answer>.
2. If a relevant detail is too small to read, request one high-resolution crop with:
<tool_call>{{"name":"request_local_region","arguments":{{"bbox_2d":[x1,y1,x2,y2]}}}}</tool_call>

For a tool call, choose the region most likely to contain the needed evidence,
including enough surrounding context to read it. bbox_2d uses integer xyxy
coordinates normalized to 0-1000 across the full low-resolution image.
The origin is top-left; x2 and y2 are exclusive.

Use the tool at most once. Do not guess unreadable details. Do not output an
answer with a tool call. Do not omit <think>, repeat either block, or write
anything outside the required tags.
"""


SECOND_PROMPT = """<image>
This is the same low-resolution full image.

<image>
This is the requested high-resolution crop.

Question: {question}

Your response MUST start with <think> and contain a non-empty </think> block.
Briefly reason from the visible evidence in these two images, especially the crop.
After </think>, output exactly <answer>short answer</answer>.

You cannot call another tool. Do not omit <think>, repeat either block, or
write anything outside the required tags.
"""
