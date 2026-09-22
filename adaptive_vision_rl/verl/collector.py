"""verl-agent trajectory collector with multi-image observations per turn."""

from __future__ import annotations

import numpy as np
import torch

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from agent_system.multi_turn_rollout.utils import process_image, torch_to_numpy
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


class AdaptiveVisionTrajectoryCollector(TrajectoryCollector):
    """Allow the answer turn to contain both the low-res image and the crop."""

    def preprocess_single_sample(self, item, gen_batch, obs):
        raw_prompt = gen_batch.non_tensor_batch["raw_prompt"][item]
        data_source = gen_batch.non_tensor_batch["data_source"][item]
        apply_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

        obs_texts = obs.get("text")
        obs_images = obs.get("image")
        obs_anchors = obs.get("anchor")
        obs_text = obs_texts[item] if obs_texts is not None else ""
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        chat = np.array([{"content": obs_text or "", "role": "user"}])
        prompt = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_kwargs,
        )
        row: dict = {}
        image_grid_thw = None

        if obs_image is not None:
            images = obs_image if isinstance(obs_image, (list, tuple)) else [obs_image]
            processed_images = [process_image(image) for image in images]
            if prompt.count("<image>") != len(processed_images):
                raise ValueError(
                    f"prompt/image mismatch: {prompt.count('<image>')} placeholders for "
                    f"{len(processed_images)} images"
                )

            raw_prompt = prompt.replace(
                "<image>", "<|vision_start|><|image_pad|><|vision_end|>"
            )
            row["multi_modal_data"] = {"image": processed_images}
            image_inputs = self.processor.image_processor(processed_images, return_tensors="pt")
            image_grid_thw = image_inputs["image_grid_thw"]
            row["multi_modal_inputs"] = dict(image_inputs)

            merge_length = self.processor.image_processor.merge_size**2
            expanded_prompt = prompt
            for grid in image_grid_thw:
                replacement = (
                    "<|vision_start|>"
                    + "<|placeholder|>" * int(grid.prod().item() // merge_length)
                    + "<|vision_end|>"
                )
                expanded_prompt = expanded_prompt.replace("<image>", replacement, 1)
            prompt = expanded_prompt.replace("<|placeholder|>", self.processor.image_token)
        else:
            raw_prompt = prompt

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.data.truncation,
        )

        if image_grid_thw is not None:
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        max_length = self.config.data.max_prompt_length
        if len(raw_prompt_ids) > max_length:
            truncation = self.config.data.truncation
            if truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-max_length:]
            elif truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:max_length]
            elif truncation == "middle":
                left = max_length // 2
                raw_prompt_ids = raw_prompt_ids[:left] + raw_prompt_ids[-(max_length - left) :]
            else:
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} exceeds max_prompt_length={max_length}"
                )

        row.update(
            {
                "input_ids": input_ids[0],
                "attention_mask": attention_mask[0],
                "position_ids": position_ids[0],
                "raw_prompt_ids": raw_prompt_ids,
                "anchor_obs": anchor,
                "index": item,
                "data_source": data_source,
            }
        )
        if self.config.data.get("return_raw_chat", False):
            row["raw_prompt"] = chat.tolist()
        return row
