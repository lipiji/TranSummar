# TranSummar
Transformer for abstractive summarization

#### cnndm:
```
---------------------------------------------
C ROUGE-1 Average_R: 0.41097 (95%-conf.int. 0.40861 - 0.41346)
C ROUGE-1 Average_P: 0.40874 (95%-conf.int. 0.40619 - 0.41141)
C ROUGE-1 Average_F: 0.39656 (95%-conf.int. 0.39451 - 0.39871)
---------------------------------------------
C ROUGE-2 Average_R: 0.17821 (95%-conf.int. 0.17590 - 0.18049)
C ROUGE-2 Average_P: 0.17781 (95%-conf.int. 0.17540 - 0.18037)
C ROUGE-2 Average_F: 0.17208 (95%-conf.int. 0.16990 - 0.17433)
---------------------------------------------
C ROUGE-3 Average_R: 0.09845 (95%-conf.int. 0.09640 - 0.10064)
C ROUGE-3 Average_P: 0.09844 (95%-conf.int. 0.09627 - 0.10069)
C ROUGE-3 Average_F: 0.09505 (95%-conf.int. 0.09307 - 0.09713)
---------------------------------------------
C ROUGE-4 Average_R: 0.06297 (95%-conf.int. 0.06109 - 0.06499)
C ROUGE-4 Average_P: 0.06329 (95%-conf.int. 0.06137 - 0.06537)
C ROUGE-4 Average_F: 0.06086 (95%-conf.int. 0.05908 - 0.06275)
---------------------------------------------
C ROUGE-L Average_R: 0.37912 (95%-conf.int. 0.37682 - 0.38160)
C ROUGE-L Average_P: 0.37726 (95%-conf.int. 0.37484 - 0.37987)
C ROUGE-L Average_F: 0.36593 (95%-conf.int. 0.36388 - 0.36810)
---------------------------------------------
C ROUGE-W-1.2 Average_R: 0.16602 (95%-conf.int. 0.16487 - 0.16715)
C ROUGE-W-1.2 Average_P: 0.27554 (95%-conf.int. 0.27362 - 0.27758)
C ROUGE-W-1.2 Average_F: 0.20031 (95%-conf.int. 0.19902 - 0.20156)
---------------------------------------------
C ROUGE-SU4 Average_R: 0.18191 (95%-conf.int. 0.17981 - 0.18403)
C ROUGE-SU4 Average_P: 0.18101 (95%-conf.int. 0.17890 - 0.18320)
C ROUGE-SU4 Average_F: 0.17496 (95%-conf.int. 0.17308 - 0.17693)
```
#### gigawords:
```
```

### How to run:
- Python 3.7, Pytorch 0.4+
- Download the processed dataset from: https://drive.google.com/file/d/1EUuEMBSlrlnf_J2jcAVl1v4owSvw_8ZF/view?usp=sharing , or you can download the original FINISHED_FILES from: https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail , and process by yourself.
- Modify the path in prepare_data.py then run it: python prepare_data.py
- Training: python -u main.py | tee train.log
- Tuning: modify main.py: is_predicting=true and model_selection=true, then run "bash tuning_deepmind.sh | tee tune.log"
- Testing: modify main.py: is_predicting=true and model_selection=false, then run "python main.py you-best-model (say cnndm.s2s.gpu4.epoch7.1)", go to "./deepmind/result/" and run  $ROUGE$ myROUGE_Config.xml C, you will get the results.
- The Perl Rouge package is enough, I did not use pyrouge.

### Reference:
- fairseq: https://github.com/pytorch/fairseq
- The Annotated Transformer: http://nlp.seas.harvard.edu/2018/04/03/attention.html

