import os
import pickle
import unittest

import jax
import requests
from tqdm import tqdm
from transformers import GPT2Tokenizer

from src.models.GPT import model_getter


class TestBasicGeneration(unittest.TestCase):
    def setUp(self) -> None:
        self.model = model_getter("unittest", return_cfg=False)

        checkpoint_files = os.listdir("checkpoints")
        if "GPT_unittest_params.pkl" not in checkpoint_files:
            # download unittest checkpoint
            save_path = f"checkpoints/GPT_unittest_params.pkl"
            ckpt_url = "https://bfattoripublic.s3.ca-central-1.amazonaws.com/models/GPT_unittest_params.pkl"
            r = requests.get(ckpt_url, stream=True)
            with open(save_path, "wb") as f:
                chunk_size = 1000
                file_size = int(r.headers["content-length"])
                with tqdm(
                    ncols=100, desc="Downloading... ", total=file_size, unit_scale=True
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(chunk_size)
                        
        with open("checkpoints/GPT_unittest_params.pkl", "rb") as f:
                self.state = pickle.load(f)

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def tearDown(self) -> None:
        pass

    def test_greedy_generation(self):
        sample_text = "This is a sample text sentence that"
        tokenized_text = self.tokenizer.encode(sample_text)
        out = self.model.generate(self.state, tokenized_text, max_length=25)
        detokenized_text_out = self.tokenizer.decode(out)
        expected_text = """This is a sample text sentence that I have not seen in a while. 
 The first thing I noticed was that the"""
        self.assertEqual(detokenized_text_out, expected_text)

    def test_temperature_generation(self):
        # ensure temperature of 1 + no sample is equivalent to greedy generation
        sample_text = "This is a sample text sentence that"
        tokenized_text = self.tokenizer.encode(sample_text)
        out = self.model.generate(
            self.state, tokenized_text, temperature=1.0, max_length=25
        )
        detokenized_text_out = self.tokenizer.decode(out)
        expected_text = """This is a sample text sentence that I have not seen in a while. 
 The first thing I noticed was that the"""
        self.assertEqual(detokenized_text_out, expected_text)

    def test_sample_generation(self):
        generation_rng = jax.random.PRNGKey(23)
        sample_text = "This is a sample text sentence that"
        tokenized_text = self.tokenizer.encode(sample_text)
        out = self.model.generate(
            self.state,
            tokenized_text,
            temperature=1.0,
            max_length=25,
            sample=True,
            sample_rng=generation_rng,
        )
        detokenized_text_out = self.tokenizer.decode(out)
        expected_text = """This is a sample text sentence that can be combined and transformed. 
 1. Writing Suarez Saan 0 Jaime 2 Joel"""
        self.assertEqual(detokenized_text_out, expected_text)

    def test_sample_generation_lower_temp(self):
        generation_rng = jax.random.PRNGKey(23)
        sample_text = "This is a sample text sentence that"
        tokenized_text = self.tokenizer.encode(sample_text)
        out = self.model.generate(
            self.state,
            tokenized_text,
            temperature=0.6,
            max_length=25,
            sample=True,
            sample_rng=generation_rng,
        )
        detokenized_text_out = self.tokenizer.decode(out)
        expected_text = """This is a sample text sentence that can be used to produce a sentence, as shown in the following screenshot. 
 <"""
        self.assertEqual(detokenized_text_out, expected_text)

    def test_generation_errors(self):
        sample_text = "This is a sample text sentence that"
        tokenized_text = self.tokenizer.encode(sample_text)
        self.assertRaises(
            AssertionError,
            self.model.generate,
            self.state,
            tokenized_text,
            25,
            0.6,
            True,
            None,
        )
