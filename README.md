# Knowledge Enhanced Sports Game Summarization


### 1. K-SportsSum Dataset
Data will be released after approval process.（Before the Spring Festival） 

### 2. Baseline Model Construction

### 3. KES model
Code will be published once the author of this repo has time. （Around conference date of WSDM'22）

### 4. Existing Works
To facilitate researchers to efficiently comprehend and follow the Sports Game Summarization task, we write a Chinese survey post: [《体育赛事摘要任务概览》](https://mp.weixin.qq.com/s/EidRYB_80AhRclz-mryVhQ), where we also discuss some future directions and give our thoughts.

We list and classify existing works of Sports Game Summarization:

| Paper | Conference/Journal | Data/Code | Category |
| :--: | :--: | :--: | :--: |
| [Towards Constructing Sports News from Live Text Commentary](https://aclanthology.org/P16-1129) | ACL 2016 | - | `Dataset`, `Ext.` |
| [Overview of the NLPCC-ICCPOL 2016 Shared Task: Sports News Generation from Live Webcast Scripts](https://link.springer.com/chapter/10.1007%2F978-3-319-50496-4_80) | NLPCC 2016 | [NLPCC 2016 shared task](http://tcci.ccf.org.cn/conference/2016/pages/page05_CFPTasks.html) | `Dataset` |
| [Research on Summary Sentences Extraction Oriented to Live Sports Text](https://link.springer.com/chapter/10.1007%2F978-3-319-50496-4_72) | NLPCC 2016 | - | `Ext.` |
| [Sports News Generation from Live Webcast Scripts Based on Rules and Templates](https://link.springer.com/chapter/10.1007%2F978-3-319-50496-4_81) | NLPCC 2016 | - | `Ext.+Temp.` |
| [Content Selection for Real-time Sports News Construction from Commentary Texts](https://aclanthology.org/W17-3504/) | INLG 2017 | - | `Ext.` |
| [Generate Football News from Live Webcast Scripts Based on Character-CNN with Five Strokes](http://csroc.org.tw/journal/JOC31-1/JOC3101-21.pdf) | 2020 | - | `Ext.+Temp.` |
| [Generating Sports News from Live Commentary: A Chinese Dataset for Sports Game Summarization](https://aclanthology.org/2020.aacl-main.61/) | AACL 2020 | [SportsSum](https://github.com/ej0cl6/SportsSum) | `Dataset`, `Ext.+Abs.` |
| [SportsSum2.0: Generating High-Quality Sports News from Live Text Commentary](https://arxiv.org/abs/2110.05750) | CIKM 2021 | [SportsSum2.0](https://github.com/krystalan/SportsSum2.0) | `Dataset`, `Ext.+Abs.` |
| [Knowledge Enhanced Sports Game Summarization](https://arxiv.org/abs/2111.12535) | WSDM 2022 | [K-SportsSum](https://github.com/krystalan/K-SportsSum) | `Dataset`, `Ext.+Abs.` |

The concepts used in Category are illustrated as follows:  
- `Dataset`: The work contributes a dataset for sports game summarization.
- `Ext.`: Extractive sports game summarization method.
- `Ext.+Temp.`: The method first extracts important commentary sentence and further utilize the human-labeled template to convey each commentary sentence to a news sentence.
- `Ext.+Abs.`: The method first extracts important commentary sentence and further utilize the seq2seq model to convey each commentary sentence to the news sentence.

### Q&A
Q1: What the differences among SportsSum, SportsSum2.0, SGSum and K-SportsSum?   
A1: **SportsSum (Huang et al. AACL 2020)** is the first large-scale Sports Game Summarization dataset which has 5428 samples. Though its wonderful contribution, the SportsSum dataset has about 15% noisy samples. Thus, **SportsSum2.0 (Wang et al, CIKM 2021)** cleans the original SportsSum and obtains 5402 samples (26 bad samples in SportsSum are removed). Following previous works, **SGSum (Non-Archival Papers, 未正式发表)** collects and cleans a large amount of data from massive games. It has 7854 samples. **K-SportsSum (Wang et al. WSDM 2022)** shuffle and randomly divide the **SGSum**. Furthermore, **K-SportsSum** has a large-scale knowledge corpus about sports teams and players, which could be useful for alleviating the knowledge gap issue (See K-SportsSum paper).

Q2: There is less code about sports game summarization.     
A2: Yeah, I know that. All existing works follow the pipeline paradigm to build sports game summarization systems. They may have two or three steps together with a pseudo label construction process. Thus, the code is too messy. For the solution, we 1) release a tutorial for building a two-step baseline for Sports Game Summarization (See Section2 in this page); 2) build an end-to-end model for public use (Work in progress, maybe will be published in 2022, but there is no guarantee. If you have experience of publishing NLP (especially NLG) papers and want to do this work with me, please feel free to write an email to me, jawang1[at]suda.edu.cn).

Q3: Any questions and suggestions?    
A3: Please feel free to contact me (jawang1[at]suda.edu.cn).

### Acknowledgement
Jiaan Wang would like to thank **[KW Lab, Fudan Univ.](http://kw.fudan.edu.cn/)** and **[iFLYTEK AI Research, Suzhou](https://www.iflytek.com/index.html)** for their helpful discussions and GPU device support.

### Citation
If you find this project is useful or use the data in your work, please consider cite our paper:
```
@article{Wang2021KnowledgeES,
  title={Knowledge Enhanced Sports Game Summarization},
  author={Jiaan Wang and Zhixu Li and Tingyi Zhang and Duo Zheng and Jianfeng Qu and An Liu and Lei Zhao and Zhigang Chen},
  journal={ArXiv},
  year={2021},
  volume={abs/2111.12535}
}
```
