from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import TokenEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class MLP(nn.Module):
    def __init__(self, d_model, input_dim, mlp_hidden_dim):
        super().__init__()

        self.linear = nn.Linear(d_model, 1)
        self.linear_classify = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, dec_out):

        x = self.linear(dec_out)
        # print(f"shape after linear: {x.shape}")
        x = x.squeeze(-1) 
        # x = x.view(1, self.seq_len, self.n_features)
        # x = dec_out.reshape(1, self.seq_len * self.n_features)
        output = self.mlp(x) 
        # print(f"shape after mlp: {output.shape}")
        output = output.squeeze()
        return output

class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.n_features = configs.n_features
        self.columns = []

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            self.llama_config.torch_dtype = torch.float16
            self.llm_model = LlamaModel.from_pretrained('huggyllama/llama-7b',config=self.llama_config)
            self.tokenizer = LlamaTokenizer.from_pretrained('huggyllama/llama-7b')

        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # for param in self.llm_model.parameters():
        #     param.requires_grad = False

        # for name, param in self.llm_model.named_parameters():
        #     print(f"{name}: {param.requires_grad}")

        self.description = 'The Medical Information Mart for Intensive Care (MIMIC) dataset is a large, de-identified and publicly-available collection of medical records.'

        self.dropout = nn.Dropout(configs.dropout)
        self.value_embedding = TokenEmbedding(1, self.d_llm)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        self.mlp = MLP(self.d_llm, configs.input_size, configs.hidden_size)

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, mask=None):
        if mask!=None:
            dec_out = self.forecast(x_enc, mask)
        dec_out = self.forecast(x_enc)
        return dec_out


    def forecast(self, x_enc, mask=None):

        # print('Start Forecasting')
        # x_enc = self.normalize_layers(x_enc, 'norm')
        # print(f'x shape: {x_enc.shape}')
        x_enc = x_enc.unsqueeze(1)
        B, T, N = x_enc.size()

        columns = ""
        for c in self.columns:
            columns = columns + "," + c
        prompt = []
        for b in range(B):
          prompt_ = (
              f"<|start_prompt|>Task description: Predict the mortality of the patient with given standard scaled data"
              f"Columns:{columns}"                
          )
          prompt.append(prompt_)

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device)) 

        x_enc = x_enc.permute(0, 2, 1)
        # # print(f'x reshape: {x_enc.shape}')
        enc_out = self.value_embedding(x_enc)
        # # print(f'x shape after value embedding: {enc_out.shape}')
        # llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        # print(f'llama input/enc_out: {llama_enc_out.shape}')
        # x_enc = self.llm_model.get_input_embeddings()(x_enc)
        dec_out = self.llm_model(inputs_embeds=enc_out).last_hidden_state        
        dec_out = dec_out[:, -N:, :]
        dec_out = self.mlp(dec_out)

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

    def set_columns(self, columns):
        self.columns = columns


