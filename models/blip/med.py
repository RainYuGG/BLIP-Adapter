import os
import torch
from lavis.models.med import BertLMHeadModel
from transformers.models.bert.configuration_bert import BertConfig

class XBertLMHeadDecoder(BertLMHeadModel):
    """
    This class decouples the decoder forward logic from the VL model.
    In this way, different VL models can share this decoder as long as
    they feed encoder_embeds as required.
    """
    
    @classmethod
    def from_config(cls, med_config_path, from_pretrained=False):
        # abs_file_path = os.path.abspath(med_config_path)
        # print(abs_file_path)
        med_config = BertConfig.from_json_file(med_config_path)

        if from_pretrained:
            return cls.from_pretrained("bert-base-uncased", config=med_config)
        else:
            return cls(config=med_config)

    def generate_from_encoder(
        self,
        tokenized_prompt,
        visual_embeds,
        sep_token_id,
        pad_token_id,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
        **kwargs
    ):

        if not use_nucleus_sampling:
            num_beams = num_beams
            visual_embeds = visual_embeds.repeat_interleave(num_beams, dim=0)

        image_atts = torch.ones(visual_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )

        model_kwargs = {
            "encoder_hidden_states": visual_embeds,
            "encoder_attention_mask": image_atts,
        }

        if use_nucleus_sampling:
            # nucleus sampling
            outputs = self.generate(
                input_ids=tokenized_prompt.input_ids,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                eos_token_id=sep_token_id,
                pad_token_id=pad_token_id,
                repetition_penalty=1.1,
                **model_kwargs
            )
        else:
            # beam search
            outputs = self.generate(
                input_ids=tokenized_prompt.input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                eos_token_id=sep_token_id,
                pad_token_id=pad_token_id,
                repetition_penalty=repetition_penalty,
                **model_kwargs
            )

        return outputs
