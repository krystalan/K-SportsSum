import os
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import get_linear_schedule_with_warmup, Adafactor
import rouge

import pytorch_lightning as pl
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel

from transformers import MBartForConditionalGeneration, MBartTokenizer, MBart50TokenizerFast

import json


class SummarizationDataset(Dataset):
    def __init__(self, split_name, tokenizer, max_input_len, max_output_len):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        ## 读取数据集，将数据集处理成 train.json, val.json 以及 test.json 三个文件夹
        with open('%s.json'%split_name, 'r', encoding='utf-8') as f:
            self.hf_dataset = json.load(f)

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        entry = self.hf_dataset[idx]
        '''
        json文件中的数据格式：
        [
            {
                "input": "一句评论句Ci1",
                "output": "对应的新闻句Ri1"
            },
            {
                "input": "一句评论句Ci2",
                "output": "对应的新闻句Ri2"
            },
            ...
            {
                "input": "一句评论句Cim",
                "output": "对应的新闻句Rim"
            },
        ]
        '''
        input_ids = self.tokenizer.encode(entry['input'].lower(), truncation=True, max_length=self.max_input_len)
        with self.tokenizer.as_target_tokenizer():
            output_ids = self.tokenizer.encode(entry['output'].lower(), truncation=True, max_length=self.max_output_len) 
        return torch.tensor(input_ids), torch.tensor(output_ids)

    @staticmethod
    def collate_fn(batch):
        pad_token_id = 1 # 对于bart/mbart来说，pad token的id是1，pegasus/t5是0
        input_ids, output_ids = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
        return input_ids, output_ids


class Summarizer(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.args = params
        self.hparams = params
        self.src_lang = "zh_CN" # 在使用mBART中，需要添加语言token，因为体育赛事摘要任务的输入与输出均为中文，所以我们都设置成中文
        self.tgt_lang = "zh_CN"
        self.tokenizer = MBart50TokenizerFast.from_pretrained(self.args.model_path, src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        self.model = MBartForConditionalGeneration.from_pretrained(self.args.model_path)
        self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None
        self.generated_id = 0

        self.decoder_start_token_id = self.tokenizer.lang_code_to_id[self.tgt_lang]
        self.model.config.decoder_start_token_id = self.decoder_start_token_id
        


    def _prepare_input(self, input_ids):
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        attention_mask[input_ids == self.tokenizer.pad_token_id] = 0
        return input_ids, attention_mask

    def forward(self, input_ids, output_ids):
        input_ids, attention_mask = self._prepare_input(input_ids)
        decoder_input_ids = output_ids[:, :-1]
        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id)

        labels = output_ids[:, 1:].clone()
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
        )

        lm_logits = outputs[0]
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))

        return [loss]

    def training_step(self, batch, batch_nb):
        output = self.forward(*batch)
        loss = output[0]
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        tensorboard_logs = {'train_loss': loss, 'lr': lr,
                            'input_size': batch[0].numel(),
                            'output_size': batch[1].numel(),
                            'mem': torch.cuda.memory_allocated(loss.device) / 1024 ** 3 if torch.cuda.is_available() else 0}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        for p in self.model.parameters():
            p.requires_grad = False

        outputs = self.forward(*batch)
        vloss = outputs[0]
        input_ids, output_ids = batch
        input_ids, attention_mask = self._prepare_input(input_ids)

        ### 这里设置inference时的长度，beam search的num等参数。
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            num_beams= 5,
            max_length = 128,
            decoder_start_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
        )
        generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        gold_str = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)
            

        return {'vloss': vloss,
                'generated': generated_str,
                'gold': gold_str
                }

    def validation_epoch_end(self, outputs):
        for p in self.model.parameters():
            p.requires_grad = True

        names = ['vloss', 'rouge1', 'rouge2', 'rougeL']
        metrics = []
        for name in names:
            metric = torch.stack([x[name] for x in outputs]).mean()
            if self.trainer.use_ddp:
                torch.distributed.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)
                metric /= self.trainer.world_size
            metrics.append(metric)
        logs = dict(zip(*[names, metrics]))
        # print(logs)


        ## 将生成的结果写入文件
        generated_str = []
        gold_str = []
        for item in outputs:
            generated_str.extend(item['generated'])
            gold_str.extend(item['gold'])

        
        with open(self.args.save_dir + '/' + self.args.save_prefix + '/generated_summary_%d.txt'%self.generated_id, 'w', encoding='utf-8') as f:
            for ending in generated_str:
                f.write(str(ending)+'\n')
        
        with open(self.args.save_dir + '/' + self.args.save_prefix + '/gold_%d.txt'%self.generated_id, 'w', encoding='utf-8') as f:
            for ending in gold_str:
                f.write(str(ending)+'\n')
        
        self.generated_id += 1
        return {'avg_val_loss': logs['vloss'], 'log': logs, 'progress_bar': logs}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        result = self.validation_epoch_end(outputs)
        # print(result)

    def configure_optimizers(self):
        if self.args.adafactor:
            optimizer = Adafactor(self.model.parameters(), lr=self.args.lr, scale_parameter=False, relative_step=False)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        num_gpus = 1 ## 设置GPU的个数, 如需使用DDP并行训练，请自行修改第301行的显卡选择。
        num_steps = self.args.dataset_size * self.args.epochs / num_gpus / self.args.grad_accum / self.args.batch_size
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup, num_training_steps=num_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_dataloader(self, current_dataloader, split_name, is_train):
        if current_dataloader is not None:
            return current_dataloader
        dataset = SummarizationDataset(split_name = split_name, tokenizer=self.tokenizer,
                                       max_input_len=self.args.max_input_len, max_output_len=self.args.max_output_len)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train) if self.trainer.use_ddp else None
        if split_name != 'train':

            return DataLoader(dataset, batch_size=self.args.val_batch_size, shuffle=(sampler is None),
                          num_workers=self.args.num_workers, sampler=sampler,
                          collate_fn=SummarizationDataset.collate_fn)
        else:
            return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=(sampler is None),
                          num_workers=self.args.num_workers, sampler=sampler,
                          collate_fn=SummarizationDataset.collate_fn)

    @pl.data_loader
    def train_dataloader(self):
        self.train_dataloader_object = self._get_dataloader(self.train_dataloader_object, 'train', is_train=True)
        return self.train_dataloader_object

    @pl.data_loader
    def val_dataloader(self):
        self.val_dataloader_object = self._get_dataloader(self.test_dataloader_object, 'val', is_train=False)
        return self.val_dataloader_object

    @pl.data_loader
    def test_dataloader(self):
        self.test_dataloader_object = self._get_dataloader(self.test_dataloader_object, 'test', is_train=False)
        return self.test_dataloader_object

    def configure_ddp(self, model, device_ids):
        model = LightningDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=False
        )
        return model

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--save_dir", type=str, default='output') # 输出文件夹
        parser.add_argument("--save_prefix", type=str, default='Sports1') # 输出文件夹，结果保存在 save_dir/save_prefix文件夹下
        parser.add_argument("--model_path", type=str, default='model/mbart-large-50-many-to-many-mmt', # mBART50文件目录
                            help="Path to the checkpoint directory or model name")
        parser.add_argument("--tokenizer", type=str, default='model/mbart-large-50-many-to-many-mmt') # mBART50文件目录
        parser.add_argument("--epochs", type=int, default=20, help="Number of epochs") # 训练epoch数
        parser.add_argument("--batch_size", type=int, default=4, help="Batch size") # batch size设置
        parser.add_argument("--val_batch_size", type=int, default=4, help="Batch size") # inference时的batch size设置
        parser.add_argument("--grad_accum", type=int, default=1, help="number of gradient accumulation steps") # 梯度累计
        parser.add_argument("--device_id", type=int, default=0, help="Number of gpus. 0 for CPU") # 使用哪一张卡做训练
        parser.add_argument("--warmup", type=int, default=500, help="Number of warmup steps")
        parser.add_argument("--lr", type=float, default=2e-5, help="Maximum learning rate") # 学习率
        parser.add_argument("--val_every", type=float, default=1.0, help="Number of training steps between validations") # 这里的意思是，没训练一个epoch，在验证集上inference一次
        parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
        parser.add_argument("--seed", type=int, default=1234, help="Seed")
        parser.add_argument("--disable_checkpointing", action='store_true', help="No logging or checkpointing")
        parser.add_argument("--max_output_len", type=int, default=128, # 输出最大长度
                            help="maximum num of wordpieces/summary. Used for training and testing")
        parser.add_argument("--max_input_len", type=int, default=1024, # 输入最大长度，请不要超过预训练模型的限制，例如mBART的最大长度限制是1024，否则会报错。
                            help="maximum num of wordpieces/summary. Used for training and testing")
        parser.add_argument("--test", action='store_true', help="Test only, no training")
        parser.add_argument("--no_progress_bar", action='store_true', help="no progress bar. Good for printing")
        parser.add_argument("--fp32", action='store_true', help="default is fp16. Use --fp32 to switch to fp32")
        parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from")
        parser.add_argument("--adafactor", action='store_true', help="Use adafactor optimizer")

        return parser


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = Summarizer(args)

    logger = TestTubeLogger(
        save_dir=args.save_dir,
        name=args.save_prefix,
        version=0  # always use version=0
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_dir, args.save_prefix, "checkpoints"),
        save_top_k=30,
        verbose=True,
        monitor='avg_val_loss',
        mode='min',
        period=-1,
        prefix=''
    )

    print(args)

    args.dataset_size = 20000  # 训练集的sample个数，会影响到warm up，请按需调整

    trainer = pl.Trainer(
        gpus = [args.device_id], 
        distributed_backend = 'ddp' if torch.cuda.is_available() else None,
        track_grad_norm = -1,
        max_epochs = args.epochs,
        replace_sampler_ddp = False,
        accumulate_grad_batches = args.grad_accum,
        val_check_interval = args.val_every,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=1,
        logger=logger,
        checkpoint_callback=checkpoint_callback if not args.disable_checkpointing else False,
        show_progress_bar=not args.no_progress_bar,
        use_amp=not args.fp32, amp_level='O2',
        resume_from_checkpoint=args.resume_ckpt,
    )
    if not args.test:
        trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="summarization")
    parser = Summarizer.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    main(args)