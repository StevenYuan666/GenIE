from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

import genie.utils.general as general_utils

log = general_utils.get_logger(__name__)


class GenieHF(T5ForConditionalGeneration):
    @classmethod
    def from_pretrained(cls, model_name_or_path, return_dict, other_parameters=None):
        #####
        # Load config
        #####
        if model_name_or_path != "random":
            config = T5Config.from_pretrained(model_name_or_path, return_dict=return_dict)
        else:
            # for the randomly initialized model we keep the same config as the genre initialized model
            config = T5Config.from_pretrained("martinjosifoski/genie-rw", return_dict=return_dict)

        if other_parameters is not None:
            for key, value in other_parameters.items():
                setattr(config, key, value)
        #####

        #####
        # Load or initialize the model
        #####
        if model_name_or_path != "random":
            # Initialization from hub or local repo
            model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, config=config)
        else:
            model = cls(config)
            log.info("Random initialization!")
        #####

        return model, config
