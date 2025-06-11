import copy
import sys
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
# 핵심 : AutoModelForCausalLM 중간 모듈을 갈아 끼우는게 허용되는가? > 아마 nn 모듈이면 안될 이유가 없을거 같다는 직관이 들긴함.

class TopkRouter(nn.Module):
    def __init__(self, n_embed:int, num_experts:int, top_k:int, dtype:torch.dtype, device:torch.device):
        super().__init__()
        self.k = top_k
        self.linear = nn.Linear(n_embed, num_experts,
                                dtype=dtype, device=device)
        #nn.init.kaiming_normal(self.linear) WIP

    def forward(self, x):                   
        logits = self.linear(x)             
        topk_logits, indices = logits.topk(self.k, dim=-1) 

        gates = torch.softmax(topk_logits, dim=-1)

        return indices, gates


class MoEFFN(nn.Module):
    def __init__(self, template_mlp, num_experts:int, top_k:int, dtype:torch.dtype, device:torch.device):
        super().__init__()
        self.hidden = template_mlp.gate_proj.in_features
        self.router = TopkRouter(self.hidden, num_experts, top_k, dtype, device)
        self.experts = nn.ModuleList([copy.deepcopy(template_mlp) # 가중치까지 복사.
                                      for _ in range(num_experts)])

    def forward(self, x):                      # x: [B, T, H]
        indices, gates = self.router(x)        # [B,T,K], [B,T,K]
        B, T, H = x.shape
        out = torch.zeros_like(x)

        # 전문가별로 토큰 모아 계산
        for e_id, expert in enumerate(self.experts):
            # K 축에서 해당 전문가가 선택된 위치 마스크
            sel_mask = (indices == e_id).any(dim=-1)
            if not sel_mask.any():
                continue

            sel = sel_mask.nonzero(as_tuple=False)
            tok = x[sel[:,0], sel[:,1]]

            # 게이트=가중치: (B,T,K) → (N,1)
            gate = gates[sel[:,0], sel[:,1],
                        (indices[sel[:,0], sel[:,1]] == e_id).nonzero(as_tuple=False)[:,1]]
            gate = gate.unsqueeze(-1)

            out_tok = expert(tok) * gate
            out[sel[:,0], sel[:,1]] += out_tok

        return out


class ChangeMoE:
    '''
    간단한 HF Wapper , 근데 Mlp 찾아서 deepcopy 로 MoE 구조로 바꿈.
    라우터는 가중치 새로 초기화. 
    '''
    def __init__(
        self,
        model_id: str,
        num_experts:int,
        top_k:int,
        dtype:torch.dtype,
        device:torch.device,
        **hf_kwargs,
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.dtype = dtype
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True, 
            **hf_kwargs,
        )
        print(f'{sys.getsizeof(self.model)}\n{self.model}')
        self._convert_all_mlps()
        if device is not None:
            self.model.to(device)
        

    def _convert_all_mlps(self):
    
        for i, layer in enumerate(self.model.model.layers):
            #print(layer)
            original_mlp = layer.mlp # < original mlp = LlamaMlp(안에 linear 3개, SiLU activa)
            layer.mlp = MoEFFN(
                template_mlp=original_mlp,
                dtype=self.dtype,
                num_experts=self.num_experts,
                top_k=self.top_k,
                device=self.device
            )
            #print(layer.mlp)
            #print(f"Layer {i}: changed FFN to MoE")

    def get_model(self)->AutoModelForCausalLM:
        return self.model
    def get_tokenizer(self)->AutoTokenizer:
        return self.tokenizer

if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_id = "meta-llama/Llama-3.2-1B"
    changer = ChangeMoE(
        model_id=model_id,
        num_experts=4,
        top_k=2,
        dtype=torch.float16,
        device=DEVICE,
    )
    moe_model =changer.get_model()
    tokenizer = changer.get_tokenizer()

    print(f'{sys.getsizeof(moe_model)}\n{moe_model}')

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    texts = ["초밥이 좋아요? 국수가 좋아요?","저녁 메뉴 추천좀요."]
    
    inputs = tokenizer(texts, return_tensors="pt",
                    padding=True, truncation=True).to(DEVICE)
    # moe_model.eval()
    # with torch.no_grad():
    #     gen_ids = moe_model.generate(
    #         input_ids       = inputs["input_ids"],
    #         attention_mask  = inputs["attention_mask"],
    #         max_new_tokens  = 64, # 생성할 토큰 수
    #         do_sample       = False # Greedy
    #     )
    moe_model.train()
    gen_ids = moe_model.generate(
        input_ids       = inputs["input_ids"],
        attention_mask  = inputs["attention_mask"],
        do_sample       = False # Greedy
    )
    decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    for i, output in enumerate(decoded):
        print(f"{i} output : {output}")
