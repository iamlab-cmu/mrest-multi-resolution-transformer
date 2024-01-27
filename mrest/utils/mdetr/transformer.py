# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
from typing import List, Optional
from omegaconf import DictConfig

import copy
import pprint

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import RobertaModel, RobertaTokenizerFast


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        pass_pos_and_query=True,
        text_encoder_type="roberta-base",
        freeze_text_encoder=False,
        contrastive_loss=False,
        use_proprio: bool = False,
        use_control_readout_emb: bool = False,
    ):
        super().__init__()

        self.pass_pos_and_query = pass_pos_and_query
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, 
                                                dropout, activation, normalize_before,)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec,
        )

        self.CLS = nn.Embedding(1, d_model) if contrastive_loss else None

        # Control readout embedding
        self.CTRL_readout = nn.Embedding(1, d_model) if use_control_readout_emb else None

        assert contrastive_loss ^ use_control_readout_emb or not (contrastive_loss and use_control_readout_emb), (
            'Cannot set both contrastive loss and control readout emb.')

        # use learned position embedding for proprio?
        self.use_proprio = use_proprio
        if use_proprio:
            self.proprio_pos_emb = nn.Embedding(1, d_model)

        self.tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type)
        self.text_encoder = RobertaModel.from_pretrained(text_encoder_type)

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        self.expander_dropout = 0.1
        config = self.text_encoder.config
        self.resizer = FeatureResizer(
            input_feat_size=config.hidden_size,
            output_feat_size=d_model,
            dropout=self.expander_dropout,
        )

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def prepare_mdetr_input(
        self,
        src=None,
        mask=None,
        query_embed=None,
        pos_embed=None,
        text=None,
        text_attention_mask=None,
        proprio_z=None,
    ):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        device = src.device
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        if self.CLS is not None:
            # We add a CLS token to the image, to be used for contrastive loss

            CLS = self.CLS.weight.view(1, 1, -1).repeat(1, bs, 1)
            # Add the CLS token to the incoming features
            src = torch.cat((CLS, src))

            # Adding zeros as the first token in the sequence to be compatible with the CLS token
            pos_embed = torch.cat((torch.zeros(1, bs, self.d_model, device=device), pos_embed))

            # Adding one mask item to the beginning of the mask to be compatible with CLS token
            cls_pad = torch.zeros(bs, 1).bool().to(device)
            mask = torch.cat((cls_pad, mask), dim=1)
        elif self.CTRL_readout is not None:
            CTRL = self.CTRL_readout.weight.view(1, 1, -1).repeat(1, bs, 1)
            # Add the CLS token to the incoming features
            src = torch.cat((CTRL, src))

            # Adding zeros as the first token in the sequence to be compatible with the CLS token
            pos_embed = torch.cat((torch.zeros(1, bs, self.d_model, device=device), pos_embed))

            # Adding one mask item to the beginning of the mask to be compatible with CLS token
            ctrl_pad = torch.zeros(bs, 1).bool().to(device)
            mask = torch.cat((ctrl_pad, mask), dim=1)


        if self.pass_pos_and_query:
            tgt = torch.zeros_like(query_embed)
        else:
            src, tgt, query_embed, pos_embed = src + 0.1 * pos_embed, query_embed, None, None

        device = src.device
        if isinstance(text[0], str):
            # Encode the text
            tokenized = self.tokenizer.batch_encode_plus(text, padding="longest", return_tensors="pt").to(device)
            encoded_text = self.text_encoder(**tokenized)
            # Transpose memory because pytorch's attention expects sequence first
            text_embd = encoded_text.last_hidden_state.transpose(0, 1)
            # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
            text_attention_mask = tokenized.attention_mask.ne(1).bool()

            # Resize the encoder hidden states to be of the same d_model as the decoder
            text_memory_resized = self.resizer(text_embd)
        else:
            # The text is already encoded, use as is.
            text_attention_mask, text_memory_resized, tokenized = text

        if proprio_z is not None and self.use_proprio:
            # Concat on the sequence dimension
            assert self.use_proprio
            proprio_z = proprio_z.unsqueeze(0)
            src = torch.cat([src, text_memory_resized, proprio_z], dim=0)

            # For mask, sequence dimension is second
            proprio_mask = torch.zeros(bs, 1).bool().to(device)
            # For mask, sequence dimension is second
            mask = torch.cat([mask, text_attention_mask, proprio_mask], dim=1)

            # Pad the pos_embed with 0 so that the addition will be a no-op for the text tokens
            proprio_pos_emb = self.proprio_pos_emb.weight.view(1, 1, -1).repeat(1, bs, 1)
            pos_embed = torch.cat([pos_embed, torch.zeros_like(text_memory_resized), proprio_pos_emb], dim=0)

        else:
            # Concat on the sequence dimension
            src = torch.cat([src, text_memory_resized], dim=0)
            # For mask, sequence dimension is second
            mask = torch.cat([mask, text_attention_mask], dim=1)
            # Pad the pos_embed with 0 so that the addition will be a no-op for the text tokens
            pos_embed = torch.cat([pos_embed, torch.zeros_like(text_memory_resized)], dim=0)

        memory_cache = {
            "src": src,
            "text_memory_resized": text_memory_resized,
            "text_pooled_op": encoded_text.pooler_output if self.CLS is not None else None,
            "mask": mask,
            "text_attention_mask": text_attention_mask,
            "pos_embed": pos_embed,
            "query_embed": query_embed,
            "tokenized": tokenized,
            "text_embedding": encoded_text.pooler_output,
        }
        return memory_cache
    
    def forward(
        self,
        src=None,
        mask=None,
        query_embed=None,
        pos_embed=None,
        text=None,
        encode_and_save=True,
        text_memory=None,
        img_memory=None,
        text_attention_mask=None,
        proprio_z=None,
        prepare_for_encoder_only: bool = False,
    ):
        if encode_and_save:
            # flatten NxCxHxW to HWxNxC, in this case bs*256*7*7 
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            device = src.device
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            mask = mask.flatten(1)

            if self.CLS is not None:
                # We add a CLS token to the image, to be used for contrastive loss

                CLS = self.CLS.weight.view(1, 1, -1).repeat(1, bs, 1)
                # Add the CLS token to the incoming features
                src = torch.cat((CLS, src))

                # Adding zeros as the first token in the sequence to be compatible with the CLS token
                pos_embed = torch.cat((torch.zeros(1, bs, self.d_model, device=device), pos_embed))

                # Adding one mask item to the beginning of the mask to be compatible with CLS token
                cls_pad = torch.zeros(bs, 1).bool().to(device)
                mask = torch.cat((cls_pad, mask), dim=1)
            elif self.CTRL_readout is not None:
                CTRL = self.CTRL_readout.weight.view(1, 1, -1).repeat(1, bs, 1)
                # Add the CLS token to the incoming features
                src = torch.cat((CTRL, src))

                # Adding zeros as the first token in the sequence to be compatible with the CLS token
                pos_embed = torch.cat((torch.zeros(1, bs, self.d_model, device=device), pos_embed))

                # Adding one mask item to the beginning of the mask to be compatible with CLS token
                ctrl_pad = torch.zeros(bs, 1).bool().to(device)
                mask = torch.cat((ctrl_pad, mask), dim=1)


            if self.pass_pos_and_query:
                tgt = torch.zeros_like(query_embed)
            else:
                src, tgt, query_embed, pos_embed = src + 0.1 * pos_embed, query_embed, None, None

            device = src.device
            if isinstance(text[0], str):
                # Encode the text
                tokenized = self.tokenizer.batch_encode_plus(text, padding="longest", return_tensors="pt").to(device)
                encoded_text = self.text_encoder(**tokenized)
                # Transpose memory because pytorch's attention expects sequence first
                text_embd = encoded_text.last_hidden_state.transpose(0, 1)
                # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
                text_attention_mask = tokenized.attention_mask.ne(1).bool()

                # Resize the encoder hidden states to be of the same d_model as the decoder
                text_memory_resized = self.resizer(text_embd)
            else:
                # The text is already encoded, use as is.
                text_attention_mask, text_memory_resized, tokenized = text

            if proprio_z is not None and self.use_proprio:
                # Concat on the sequence dimension
                assert self.use_proprio
                proprio_z = proprio_z.unsqueeze(0)
                src = torch.cat([src, text_memory_resized, proprio_z], dim=0)

                # For mask, sequence dimension is second
                proprio_mask = torch.zeros(bs, 1).bool().to(device)
                # For mask, sequence dimension is second
                mask = torch.cat([mask, text_attention_mask, proprio_mask], dim=1)

                # Pad the pos_embed with 0 so that the addition will be a no-op for the text tokens
                proprio_pos_emb = self.proprio_pos_emb.weight.view(1, 1, -1).repeat(1, bs, 1)
                pos_embed = torch.cat([pos_embed, torch.zeros_like(text_memory_resized), proprio_pos_emb], dim=0)

            else:
                # Concat on the sequence dimension
                src = torch.cat([src, text_memory_resized], dim=0)
                # For mask, sequence dimension is second
                mask = torch.cat([mask, text_attention_mask], dim=1)
                # Pad the pos_embed with 0 so that the addition will be a no-op for the text tokens
                pos_embed = torch.cat([pos_embed, torch.zeros_like(text_memory_resized)], dim=0)
                
            # Here src = [58, bs, 256] where 256 is number of channels (taken as embedding_dim), img_memory = [58, bs, 256]
            img_memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

            if proprio_z is not None and self.use_proprio:
                text_memory_start_idx = img_memory.size(0) - (1 + len(text_memory_resized))
                text_memory = img_memory[-(1 + len(text_memory_resized)):-1]
            else:
                text_memory_start_idx = img_memory.size(0) - len(text_memory_resized)
                text_memory = img_memory[-len(text_memory_resized) :]

            assert img_memory.shape[1] == text_memory.shape[1] == tgt.shape[1]
            memory_cache = {
                "text_memory_resized": text_memory_resized,
                "text_memory_start_idx": text_memory_start_idx,
                "text_memory": text_memory,
                "img_memory": img_memory,
                "text_pooled_op": encoded_text.pooler_output if self.CLS is not None else None,
                "img_pooled_op": img_memory[0] if self.CLS is not None else None,  # Return the CLS token
                "ctrl_pooled_op": img_memory[0] if self.CTRL_readout is not None else None,  # Return the CLS token
                "mask": mask,
                "text_attention_mask": text_attention_mask,
                "pos_embed": pos_embed,
                "query_embed": query_embed,
                "tokenized": tokenized,
                "text_embedding": encoded_text.pooler_output,
            }
            return memory_cache

        else:
            if self.pass_pos_and_query:
                tgt = torch.zeros_like(query_embed)
            else:
                src, tgt, query_embed, pos_embed = src + 0.1 * pos_embed, query_embed, None, None

            assert img_memory.shape[1] == text_memory.shape[1] == tgt.shape[1]

            hs = self.decoder(
                tgt,
                img_memory,
                text_memory,
                memory_key_padding_mask=mask,
                text_memory_key_padding_mask=text_attention_mask,
                pos=pos_embed,
                query_pos=query_embed,
            )
            return hs.transpose(1, 2)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None,):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):

        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                text_memory=text_memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                text_memory_key_padding_mask=text_memory_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation="relu", normalize_before=False,):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.d_model = d_model
        self.dropout_rate = dropout
        self.nhead = nhead

        self.multiview_cross_attn_config = None
    
    def create_multiview_models(self, multiview_cross_attn_config: DictConfig):
        self.multiview_cross_attn_config = multiview_cross_attn_config

        dim = self.d_model
        attn_drop = multiview_cross_attn_config.attn_drop
        proj_drop = multiview_cross_attn_config.proj_drop
        bias = True

        self.qkv_hand_cam1to2 = nn.Linear(dim, dim * 2, bias=bias)
        self.qkv_cam1to2 = nn.Linear(dim, dim, bias=bias)
        self.attn_drop_cam1to2 = nn.Dropout(attn_drop)
        self.proj_cam1to2 = nn.Linear(dim, dim)
        self.proj_drop_cam1to2 = nn.Dropout(proj_drop)
        self.norm_cam1to2 = nn.LayerNorm(dim)

        self.softmax = nn.Softmax(dim=-1)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        src_other_cam: Optional[Tensor] = None,
        return_attn_weights: bool = False,
    ):
        attn_weights_dict = dict()

        q = k = self.with_pos_embed(src, pos)
        if return_attn_weights:
            src2, self_attn_weights = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
            attn_weights_dict['self_attn_weights'] = self_attn_weights
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src2 = self.dropout1(src2)

        src = src + src2

        if (src_other_cam is not None and
            self.multiview_cross_attn_config is not None and
            self.multiview_cross_attn_config.get('use_with_self_attn', False)):
            N_cam2, B_cam2, C_cam2 = src_other_cam.shape
            kv_cam2 = (
                self.qkv_hand_cam1to2(src_other_cam)
                .reshape(N_cam2, B_cam2, 2, self.nhead, C_cam2 // self.nhead)
                .permute(2, 1, 3, 0, 4)
            )
            k_cam2, v_cam2 = kv_cam2[0], kv_cam2[1]

            N_cam1, B_cam1, C_cam1 = src.shape
            q_cam1 = self.qkv_cam1to2(src)
            q_cam1 = q_cam1.reshape(N_cam1, B_cam1, 1, self.nhead, C_cam1 // self.nhead).permute(
                2, 1, 3, 0, 4)
            q_cam1 = q_cam1[0]

            sqrt_D = k_cam2.size(-1) ** -0.5
            attn_cam1to2 = (q_cam1 * sqrt_D) @ k_cam2.transpose(-2, -1)  # B, nH, N_cam1, N_cam2

            if return_attn_weights:
                attn_weights_dict['cam1to2_attn_weights'] = self.softmax(attn_cam1to2)

            # Should we add something to the attn part (we may want to optionally mask things from cam2)

            attn_cam1to2 = self.attn_drop_cam1to2(self.softmax(attn_cam1to2))
            y = (attn_cam1to2 @ v_cam2).transpose(1, 2)
            y = y.reshape(B_cam1, N_cam1, C_cam1).transpose(1, 0)
            y = self.proj_drop_cam1to2(self.proj_cam1to2(y))
            src = src + y

        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.dropout2(src2)

        src = src + src2

        src = self.norm2(src)

        if (src_other_cam is not None and
            self.multiview_cross_attn_config is not None and
            self.multiview_cross_attn_config.get('use_after_ffw', False)):
            assert self.multiview_cross_attn_config is not None

            assert not self.multiview_cross_attn_config.get('use_with_self_attn', False), (
                f'Incorrect params cannot use both with self-attn and after FFW.')

            N_cam2, B_cam2, C_cam2 = src_other_cam.shape
            kv_cam2 = (
                self.qkv_hand_cam1to2(src_other_cam)
                .reshape(N_cam2, B_cam2, 2, self.nhead, C_cam2 // self.nhead)
                .permute(2, 1, 3, 0, 4)
            )
            k_cam2, v_cam2 = kv_cam2[0], kv_cam2[1]

            N_cam1, B_cam1, C_cam1 = src.shape
            q_cam1 = self.qkv_cam1to2(src)
            q_cam1 = q_cam1.reshape(N_cam1, B_cam1, 1, self.nhead, C_cam1 // self.nhead).permute(
                2, 1, 3, 0, 4)
            q_cam1 = q_cam1[0]

            sqrt_D = k_cam2.size(-1) ** -0.5
            attn_cam1to2 = (q_cam1 * sqrt_D) @ k_cam2.transpose(-2, -1)  # B, nH, N_cam1, N_cam2

            if return_attn_weights:
                attn_weights_dict['cam1to2_attn_weights'] = self.softmax(attn_cam1to2)

            # Should we add something to the attn part (we may want to optionally mask things from cam2)

            attn_cam1to2 = self.attn_drop_cam1to2(self.softmax(attn_cam1to2))
            y = (attn_cam1to2 @ v_cam2).transpose(1, 2)
            y = y.reshape(B_cam1, N_cam1, C_cam1).transpose(1, 0)
            y = self.proj_drop_cam1to2(self.proj_cam1to2(y))

            # y = src + y
            src = src + y

        if return_attn_weights:
            return src, attn_weights_dict
        else:
            return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        src_other_cam: Optional[Tensor] = None,
        return_attn_weights: bool = False,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)

        if return_attn_weights:
            src2, attn_weights = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        else:
            src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]

        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))

        src = src + self.dropout2(src2)

        if return_attn_weights:
            return src, attn_weights
        else:
            return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        src_other_cam: Optional[Tensor] = None,
        return_attn_weights: bool = False,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, src_other_cam=src_other_cam,
                                    return_attn_weights=return_attn_weights)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, src_other_cam=src_other_cam,
                                 return_attn_weights=return_attn_weights)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 normalize_before=False,):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.cross_attn_text = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
    
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # For now, trying one version where its self attn -> cross attn text -> cross attn image -> FFN
    def forward_post(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)

        # Self attention
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention to text
        # tgt2 = self.cross_attn_text(
        #     query=self.with_pos_embed(tgt, query_pos),
        #     key=text_memory,
        #     value=text_memory,
        #     attn_mask=None,
        #     key_padding_mask=text_memory_key_padding_mask,
        # )[0]
        # tgt = tgt + self.dropout2(tgt2)
        # tgt = self.norm2(tgt)

        # Cross attention to image
        tgt2 = self.cross_attn_image(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        assert False, "not implemented yet"
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos
            )
        return self.forward_post(
            tgt,
            memory,
            text_memory,
            tgt_mask,
            memory_mask,
            text_memory_key_padding_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    transformer = Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        pass_pos_and_query=args.pass_pos_and_query,
        text_encoder_type=args.text_encoder_type,
        freeze_text_encoder=args.freeze_text_encoder,
        contrastive_loss=args.contrastive_loss,
        # NOTE: This is not in the original MDETR
        use_proprio=args.use_proprio,
        use_control_readout_emb=args.use_control_readout_emb,
    )
    if not args.train_transformer:
        transformer_train_params = []
        transformer_notrain_params = []
        for name, param in transformer.named_parameters():
            if 'text_encoder' in name:
                requires_grad = False
            elif 'adapter' in name:
                requires_grad = True
            elif 'resizer' in name and args.train_resizer:
                requires_grad = True
            elif 'norm' in name and args.train_transformer_layernorm:
                requires_grad = True
            elif 'CTRL_readout' in name:
                assert name == 'CTRL_readout.weight'
                requires_grad = True
            else:
                requires_grad = False
            param.requires_grad_(requires_grad)
            if requires_grad:
                transformer_train_params.append(name)
            else:
                transformer_notrain_params.append(name)
        print('====> Will train transformer params')
        pprint.pprint(transformer_train_params)
        print('====> Will NOT train transformer params')
        pprint.pprint(transformer_notrain_params)

    return transformer


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
