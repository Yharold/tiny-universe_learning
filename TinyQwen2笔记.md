# Qwen2学习笔记

跟着tiny-universe这个项目打算把一些模型都写一遍。首先写的就是Qwen2，这是记录Qwen2的学习内容。

## 主模型类：Qwen2Model
Qwen的主题架构如下图所示：

![框架图](./img/framework.JPEG)  

其中的主架构就分为三部分：
- tokenizer:将文本转为词表索引的类
- Embedding：将索引转为对应向量的类
- Layers:最主要的部分，相当于Transformer的解码器，有很多个Qwen2Decoder类
- RMSNorm：一个规范化方法

剩下的Linear，Loss，Output其实就不算模型内的了，它会根据训练和推理的不同而选择不同的处理办法。

先看主模型的代码：

```python
from torch import nn
# Qwen2的主模型
class Qwen2Model(nn.Model):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(self.config.vocab_size,self.config.hidden_size,self.config.padding_idx)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config,layer_idx)
                                     for layer_idx in range(self.config.num_hidden_layers)])
        self.norm = Qwen2RMSNorm(self.config.hidden_size,eps=self.config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()
    def post_init(self):
        self.init_weights()
        self._backward__compatibility_gradient_checkpointing()
        
    
    def forward(self,input_ids,position_ids):
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        if self.config.output_hidden_states:
            all_hidden_states = ()
        for idx,decoder_layer in enumerate(self.layers):
            if self.config.output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask = self.config.attention_mask,
                position_ids = position_ids,
                past_key_value = self.config.past_key_value,
                output_attentions = self.config.output_attentions,
                use_cache = self.config.use_cache,
            )
            hidden_states = layer_outputs[0]
        hidden_states = self.norm(hidden_states)    
        if self.config.output_hidden_states:
            all_hidden_states += (hidden_states,)
            return all_hidden_states
        else:
            return hidden_states
```
主模型中的的初始化参数就一个config，其实这也是一个类，是Qwen2Config类。基本上用的到的参数都写在了这个类中。所以想要了解Qwen2需要的参数，最好去看这个类。当然了，是看完其他知识点后再去看，不然看到参数什么也不懂。

主模型初始化中只创建了三个类：Embedding，layers和norm,分别对应Embedding，layers和RMSNorm。

Embedding需要一个词表大小vocab_size和词元维度dim，还有一个可选的填充字符在词表中的索引padding_idx。
layers本身其实就是一个数组，里面存放了很多个Qwen2Decoder类
RMSNorm就是一个简单的规范化类

整个的执行逻辑就是一个简单的线性执行。输入input_ids，即文本词元在词表对应的索引。然后执行Embedding，接着就是Qwen2Decoders，最后一个规范化RMSNorm就行了。

## 解码器类：Qwen2Decoder

Qwen2Decoder类是Qwen2的核心类。它主要包括四个类：

- self_attn:Attention类，用来执行Attention，默认选的是Qwen2Attention
- mlp:一个MLP，主要就是全连接层
- input_layernorm:规范化类
- post_attention_layer:规范化类

整个模型的结构图如下：

![Qwen2Decoder](./img/decoderlayer.png)

整个结构可以说分为两个部分：第一部分就是RMSNorm+Attention+residual；第二部分就是RMSNorm+mlp+residual。也就是说这两部分也就attention和mlp不同。

代码如下：
```python
QWEN2_ATTENTION_CLASSES = {
    "eager":Qwen2Attention, 
    "flah_attention_2":Qwen2FlashAttention2,
    "sdpa":Qwen2SdpaAttention,
}        
class Qwen2DecoderLayer(nn.Module):
    def __init__(self,config,layer_idx):
        super().__init__()
        self.config = config
        self.self_attn =  QWEN2_ATTENTION_CLASSES[config._attn_implementation](config,layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        
    def forward(self,hidden_states,position_ids,attention_mask,past_key_value,output_attentions):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states,self_attn_weights,\
        present_key_value= self.self_attn(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_value = past_key_value,
            output_attentions = output_attentions,
            use_cache = self.config.use_cache,
        )
        hidden_states = hidden_states+residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states+residual
        return hidden_states
```
## 注意力类：Qwen2Attention

结构图如下：

![Qwen2Attention](./img/Qwen2Attention.png)

这是整个Qwen2的重点。

首先，它使用的是旋转位置编码，这是一种可以用绝对位置编码方式来实现相对位置编码方式的方法，最大的优点就是不影响attention的优化。因为它是对q，k分别执行位置编码，而不是对$\frac{q*k^T}{\sqrt{d}}$执行编码。

其次，它使用的不是传统的多头注意力，而是GQA，这会大大降低训练的运算量和内存压力，特别是KV缓存的压力。

所以整个程序的重点就两个，一个就是实现GQA，另一个就是实现旋转位置编码RoPE。

最后吐槽一下这个图，里面有一个Repeat_kv，这个块实际处理的是Key和Value，所以叫做Repeat_kv，但图上画的却是从Query和Key引下来的，给人感觉是对Query和Key执行Repeat_kv。

模型的初始化中主要就四个线性类nn.Linear用来得到Query，Key，Value以及将结果投影回来，另外还有一个Qwen2RotaryEmbedding用来得到位置编码矩阵sin和cos。

整个模型的执行顺序没什么说的。

1. 先执行输入的投影，即计算query，key，value
2. 对query，key，value的形状进行改变，即从原本的[bs,q_len,num_heads*head_dim]转为[bs,num_heads,q_len,head_dim]
3. 计算旋转位置矩阵sin，cos
4. 使用sin，cos对query，key执行位置编码，得到编码后的query和key
5. 对key，value执行repeat_kv,使得key，value的num_heads与query相等
6. 执行公式$\frac{q*k^T}{\sqrt{d}}$，得到attn_weights
7. 对attn_weights执行mask操作
8. 执行softmax函数，
9. 接着执行dropout
10. 然后与value相乘，得到attn_output
11. 接着修改attn_output的形状
12. 最后将attn_output投影回去，得到最终的结果
代码如下：
```python
class Qwen2Attention(nn.Module):
    def __init__(self,config,layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_causal = True
        if(self.config.head_dim * self.config.num_heads) != self.config.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads!(got 'hidden_size:'{self.config.hidden_size} and 'num_heads:'{self.config.num_heads})")
        self.q_proj = nn.Linear(self.config.hidden_size,self.config.num_heads*self.config.head_dim,bias=self.config.attention_bias)
        self.k_proj = nn.Linear(self.config.hidden_size,self.config.num_key_value_heads*self.config.head_dim,bias=self.config.attention_bias)
        self.v_proj = nn.Linear(self.config.hidden_size,self.config.num_key_value_heads*self.config.head_dim,bias=self.config.attention_bias)
        self.o_proj = nn.Linear(self.config.num_heads*self.config.head_dim,self.config.hidden_size,bias=self.config.attention_bias)
        self.rotary_emb = Qwen2RotaryEmbedding(self.config.head_dim,
                    max_position_embeddings=self.config.max_position_embeddings,base=self.config.rope_theta)
    def forward(self,hidden_states,attention_mask):
        bsz,q_len,_=hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz,q_len,self.config.num_heads,self.config.head_dim).transpose(1,2)
        key_states = key_states.view(bsz,q_len,self.config.num_key_value_heads,self.config.head_dim).transpose(1,2)
        value_states = value_states.view(bsz,q_len,self.config.key_value_heads,self.config.head_dim).transpose(1,2)
        cos,sin = self.rotary_emb(value_states,seq_len=self.config.kv_seq_len)
        query_states,key_states = apply_rotary_pos_emb(query_states,key_states,cos,sin,self.config.position_ids)
        key_states = repeat_kv(key_states,self.config.num_key_value_groups)
        value_states = repeat_kv(value_states,self.config.num_key_value_groups)
        attn_weights = torch.matmul(query_states,key_states.transpose(2,3))/math.sqrt(self.config.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights,dim=-1,dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights,p=self.config.attention_dropout,training=self.config.training)
        attn_output = torch.matmul(attn_weights,value_states)
        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output = attn_output.reshape(bsz,q_len,self.config.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output
```


## 旋转位置编码矩阵：Qwen2RotaryEmbedding

旋转位置编码RoPE我这里不作过多的展开讲解，因为确实比较复杂。我建议大家可以看看这篇博客
[Transformer升级之路：2、博采众长的旋转式位置编码](https://spaces.ac.cn/archives/8265)

创始大佬亲自写的科普文章就问你看不看！！！（苏大佬真的是一个低调的扫地僧，这个科学空间的文章都值得一看）

整个论文的思路过程就是求一个编码函数$\bar{q}_m=f(q,m)$可以对输入的x加上位置m的信息，使其成为$bar{q}_m$。这里的m就是指x在序列中的位置。


公式9给出了这个函数的形式，推理的过程就不讲了（我不觉得我会比博客讲的更好）。这里的q就是token本身，是一个向量。公式10则是给出了当q是一个二维向量时的矩阵形式。我这里理解错了，一开始把q0理解成了第一个token，q1理解成了第二个token。q0是q的第一个维度，q1是第二个维度。纠正了这点这个公式就不难理解了。

公式11则给出了q是d维时的矩阵形式。公式13则是一个简化计算的形式。

总之，按照公式13给token $q$加上位置编码。其中的位置矩阵有两个，sin和cos。这也是Qwen2RotaryEmbedding的主要功能----求取位置编码矩阵。

实际的执行其实并不是按照公式13执行的，具体可以看apply_rotary_pos_emb，文字描述好难。具体可以看下图
![](./img/ROPE3.png)
这个的原理个人感觉应该是q本身是无序的，也就是$q_0,q_1,q_2,...$和$q_0,q_2,q_5...$不应该有区别。所以它可以任意调换位置。

代码如下：


```python
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self,dim,max_position_embeddings=2048,base=10000,device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # 生成角度theta
        inv_freq = 1.0/(self.base)**(torch.arange(0,self.dim,2,dtype=torch.int64).float().to(device)/self.dim)
        # 定义一组参数，训练时不会更新，即不会因为optimizer.step()改变（除非动手改变）
        # 但在保存模型时会保存这组数据
        self.register_buffer("inv_freq",inv_freq,persistent=False)

        self._set_cos_sin_cache(seq_len=max_position_embeddings,device=self.inv_freq.device,dtype=torch.get_default_dtype())
    
    def _set_cos_sin_cache(self,seq_len,device,dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached,device=device,dtype=torch.int64).type_as(self.inv_freq)
        # 计算序列序号与角度的内积，即sin和cos内的值
        freqs = torch.outer(t,self.inv_freq)
        # 因为奇偶是分开的，所以这里重复了一下，这样emb就和x的维度相同了
        emb = torch.cat((freqs,freqs),dim=-1)
        # 计算cos和sin，并注册数据
        self.register_buffer("cos_cached",emb.cos().to(dtype),persistent=False)
        self.register_buffer("sin_cached",emb.sin().to(dtype),persistent=False)
    
    def forward(self,x:torch.Tensor,seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len,device=x.device,dtype=x.dtype)
        return (self.cos_cached[:seq_len].to(dtype=x.dtype),
                self.sin_cached[:seq_len].to(dtype=x.dtype))  

def rotate_half(x):
    x1 = x[...,:x.shape[-1]//2]
    x2 = x[...,x.shape[-1]//2:]
    return torch.cat((-x2,x1),dim=-1)

def apply_rotary_pos_emb(q,k,cos,sin,position_ids,unsqueeze_dim=1):
    
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q*cos)+(rotate_half(q)*sin)
    k_embed = (k*cos)+(rotate_half(k)*sin)
    return q_embed,k_embed
```

## 全连接：Qwen2MLP

这个全连接类其实没什么讲的，结构图如下：
![MLP](./img/MLP1.png)

整个MLP的初始化中主要包含了三个线性层和一个激活函数(Act)。这个MLP没有使用线性连接，而是通过一个线性层和激活函数组成了一个门阈结构。

```python
class Qwen2MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size,self.intermediate_size,bias=False)
        self.up_proj = nn.Linear(self.hidden_size,self.intermediate_size,bias=False)
        self.down_proj = nn.Linear(self.hidden_size,self.intermediate_size,bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
    def forward(self,x:torch.Tensor):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x))
        return down_proj

```
## 规范化：Qwen2RMSNorm

这个类就更没什么说的了，就是执行了一个规范化。和一般的规范化不同的是它只有一个权重weight，没有偏置bias。公式就是
$RMSNorm(x)=\frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}w_{i}^{2}+\epsilon}}$

```python
class Qwen2RMSNorm(nn.Module):
    def __init__(self,hidden_size,eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    def forward(self,hidden_states:torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1,keepdim=True)
        hidden_states = hidden_states*torch.rsqrt(variance+self.variance_epsilon)
        return self.weight*hidden_states.to(input_dtype)   
```


