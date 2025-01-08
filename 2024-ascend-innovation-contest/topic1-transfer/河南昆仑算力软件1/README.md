# Mixtral优化方法：

## 1、	使用mint替换ops

```
        causal_mask *= ops.arange(target_length) > cache_position.reshape(-1, 1)
```

改为：

```
        causal_mask *= mint.arange(target_length) > cache_position.reshape(-1, 1)
```

收益较小，平均1-2ms的提升

## 2、	中括号方式的索引为profiler中主要异常来源，可替换为index_select或者narrow或者其他方式，部分替换如下

源码：

```
padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
```

改为：

```
 padding_mask = mint.narrow(causal_mask, -1, 0, mask_length) + attention_mask[:, None, None, :]
```



源码：

（attention_mask修改后更为复杂，实测部分修改性能更好）

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
改为

```
        return (
            mint.narrow(self.cos_cached ,0 ,0, seq_len).to(dtype=x.dtype),
            mint.narrow(self.sin_cached ,0 ,0, seq_len).to(dtype=x.dtype),
        )
```



源码：

        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
    return ops.cat((-x2, x1), dim=-1)
改为：

```
    x1 ,x2 =mint.split(x,x.shape[-1] //2, dim = -1)
return mint.cat((-x2, x1), dim=-1)
```



源码：

```
cos = cos[position_ids].unsqueeze(unsqueeze_dim)
sin = sin[position_ids].unsqueeze(unsqueeze_dim)
```

改为：

```
cos = F.embedding(position_ids, cos).unsqueeze(unsqueeze_dim)
sin = F.embedding(position_ids, sin).unsqueeze(unsqueeze_dim)

```



源码：

            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
改为

```
        split_size = key_states.shape[-2] 
        splits = mint.split(attention_mask, [split_size, attention_mask.shape[-1] - split_size], -1)  
        causal_mask = splits[0] 
```


以上方式可消除profiler中明显空洞的地方，改为亲和npu的写法后，消除平均20ms左右的时间 

## 3、	Moe优化，将nonzero一次申请转换为np类型，后续填充

源码：

```
        expert_mask = nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = ops.nonzero(expert_mask[expert_idx], as_tuple=True)

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            if 0 not in idx.shape:
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

                # However `index_add_` only support torch tensors for indexing so we'll use
                # the `top_x` tensor here.
                final_hidden_states = final_hidden_states.index_add(0, top_x.int(), current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
```

改为：

```
        non_zero_df = mint.nonzero(expert_mask).asnumpy()
        if non_zero_df.shape[0] > 0:
            expert_idx = non_zero_df[0][0]
            j = 0
            k = 1
            for i in range(1, non_zero_df.shape[0]):
                if non_zero_df[i][0] == expert_idx:
                    k += 1
                else:
                    expert_layer = self.experts[expert_idx]
                    idx = mindspore.Tensor(non_zero_df[j:j + k, 1], mindspore.int32)
                    top_x = mindspore.Tensor(non_zero_df[j:j + k, 2], mindspore.int32)
                    current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                    current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
                    final_hidden_states = final_hidden_states.index_add(0, top_x.int(), current_hidden_states.to(hidden_states.dtype))
                    expert_idx = non_zero_df[i][0]
                    j = i
                    k = 1
            expert_layer = self.experts[expert_idx]
            idx = mindspore.Tensor(non_zero_df[j:j + k, 1], mindspore.int32)
            top_x = mindspore.Tensor(non_zero_df[j:j + k, 2], mindspore.int32)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)

            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states = final_hidden_states.index_add(0, top_x.int(), current_hidden_states.to(hidden_states.dtype))
```



## 4、	使用RMSnorm融合算子替换小算子

源码：

```
        variance = ops.mean(hidden_states.pow(2), -1, keepdim=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```

改为：

```
        if not self.training and use_pyboost() and not ON_ORANGE_PI:
            return F.rms_norm(hidden_states, self.weight, self.variance_epsilon)
```

