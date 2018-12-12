# TranSummar
Transformer for abstractive summarization

#### v0.1, pure transformer (no copy, no coverage), God...
```
---------------------------------------------
C ROUGE-1 Average_R: 0.26888 (95%-conf.int. 0.26473 - 0.27306)
C ROUGE-1 Average_P: 0.30623 (95%-conf.int. 0.30112 - 0.31147)
C ROUGE-1 Average_F: 0.27602 (95%-conf.int. 0.27224 - 0.27986)
---------------------------------------------
C ROUGE-2 Average_R: 0.06586 (95%-conf.int. 0.06360 - 0.06822)
C ROUGE-2 Average_P: 0.07486 (95%-conf.int. 0.07230 - 0.07753)
C ROUGE-2 Average_F: 0.06749 (95%-conf.int. 0.06526 - 0.06979)
---------------------------------------------
C ROUGE-3 Average_R: 0.01741 (95%-conf.int. 0.01601 - 0.01914)
C ROUGE-3 Average_P: 0.01972 (95%-conf.int. 0.01826 - 0.02153)
C ROUGE-3 Average_F: 0.01780 (95%-conf.int. 0.01650 - 0.01949)
---------------------------------------------
C ROUGE-4 Average_R: 0.00570 (95%-conf.int. 0.00471 - 0.00704)
C ROUGE-4 Average_P: 0.00639 (95%-conf.int. 0.00535 - 0.00777)
C ROUGE-4 Average_F: 0.00582 (95%-conf.int. 0.00485 - 0.00718)
---------------------------------------------
C ROUGE-L Average_R: 0.24405 (95%-conf.int. 0.24035 - 0.24779)
C ROUGE-L Average_P: 0.27773 (95%-conf.int. 0.27302 - 0.28243)
C ROUGE-L Average_F: 0.25045 (95%-conf.int. 0.24704 - 0.25405)
---------------------------------------------
C ROUGE-W-1.2 Average_R: 0.10503 (95%-conf.int. 0.10343 - 0.10673)
C ROUGE-W-1.2 Average_P: 0.19818 (95%-conf.int. 0.19510 - 0.20140)
C ROUGE-W-1.2 Average_F: 0.13228 (95%-conf.int. 0.13048 - 0.13407)
---------------------------------------------
C ROUGE-SU4 Average_R: 0.08701 (95%-conf.int. 0.08492 - 0.08903)
C ROUGE-SU4 Average_P: 0.09939 (95%-conf.int. 0.09709 - 0.10182)
C ROUGE-SU4 Average_F: 0.08906 (95%-conf.int. 0.08716 - 0.09113)
```

#### OpenNMT (no copy, no coverage), good...
```
---------------------------------------------
C ROUGE-1 Average_R: 0.33596 (95%-conf.int. 0.33384 - 0.33812)
C ROUGE-1 Average_P: 0.42363 (95%-conf.int. 0.42103 - 0.42636)
C ROUGE-1 Average_F: 0.36314 (95%-conf.int. 0.36123 - 0.36515)
---------------------------------------------
C ROUGE-2 Average_R: 0.12626 (95%-conf.int. 0.12458 - 0.12800)
C ROUGE-2 Average_P: 0.16142 (95%-conf.int. 0.15935 - 0.16360)
C ROUGE-2 Average_F: 0.13715 (95%-conf.int. 0.13540 - 0.13902)
---------------------------------------------
C ROUGE-3 Average_R: 0.06299 (95%-conf.int. 0.06160 - 0.06443)
C ROUGE-3 Average_P: 0.08174 (95%-conf.int. 0.07981 - 0.08352)
C ROUGE-3 Average_F: 0.06878 (95%-conf.int. 0.06726 - 0.07028)
---------------------------------------------
C ROUGE-4 Average_R: 0.03680 (95%-conf.int. 0.03573 - 0.03797)
C ROUGE-4 Average_P: 0.04861 (95%-conf.int. 0.04711 - 0.05017)
C ROUGE-4 Average_F: 0.04042 (95%-conf.int. 0.03922 - 0.04164)
---------------------------------------------
C ROUGE-L Average_R: 0.30994 (95%-conf.int. 0.30802 - 0.31202)
C ROUGE-L Average_P: 0.39101 (95%-conf.int. 0.38849 - 0.39371)
C ROUGE-L Average_F: 0.33511 (95%-conf.int. 0.33327 - 0.33713)
---------------------------------------------
C ROUGE-W-1.2 Average_R: 0.13446 (95%-conf.int. 0.13350 - 0.13544)
C ROUGE-W-1.2 Average_P: 0.28430 (95%-conf.int. 0.28240 - 0.28625)
C ROUGE-W-1.2 Average_F: 0.17691 (95%-conf.int. 0.17587 - 0.17803)
---------------------------------------------
C ROUGE-SU4 Average_R: 0.13850 (95%-conf.int. 0.13693 - 0.14006)
C ROUGE-SU4 Average_P: 0.17830 (95%-conf.int. 0.17631 - 0.18030)
C ROUGE-SU4 Average_F: 0.15050 (95%-conf.int. 0.14891 - 0.15210)
```

#### OpenNMT+copy, good...
```
---------------------------------------------

```

#### v0.1 pure transformer (no copy, no coverage), agiga dataset:
```
C ROUGE-1 Average_R: 0.31383 (95%-conf.int. 0.30249 - 0.32486)
C ROUGE-1 Average_P: 0.34679 (95%-conf.int. 0.33447 - 0.35839)
C ROUGE-1 Average_F: 0.31882 (95%-conf.int. 0.30743 - 0.32919)
---------------------------------------------
C ROUGE-2 Average_R: 0.13815 (95%-conf.int. 0.12913 - 0.14802)
C ROUGE-2 Average_P: 0.15193 (95%-conf.int. 0.14248 - 0.16191)
C ROUGE-2 Average_F: 0.13939 (95%-conf.int. 0.13044 - 0.14882)
---------------------------------------------
C ROUGE-3 Average_R: 0.07131 (95%-conf.int. 0.06336 - 0.07958)
C ROUGE-3 Average_P: 0.07810 (95%-conf.int. 0.06958 - 0.08702)
C ROUGE-3 Average_F: 0.07146 (95%-conf.int. 0.06367 - 0.07970)
---------------------------------------------
C ROUGE-4 Average_R: 0.03936 (95%-conf.int. 0.03270 - 0.04626)
C ROUGE-4 Average_P: 0.04412 (95%-conf.int. 0.03668 - 0.05157)
C ROUGE-4 Average_F: 0.03953 (95%-conf.int. 0.03304 - 0.04650)
---------------------------------------------
C ROUGE-L Average_R: 0.28961 (95%-conf.int. 0.27922 - 0.30049)
C ROUGE-L Average_P: 0.32052 (95%-conf.int. 0.30914 - 0.33166)
C ROUGE-L Average_F: 0.29444 (95%-conf.int. 0.28413 - 0.30491)
---------------------------------------------
C ROUGE-W-1.2 Average_R: 0.18007 (95%-conf.int. 0.17335 - 0.18725)
C ROUGE-W-1.2 Average_P: 0.29822 (95%-conf.int. 0.28764 - 0.30886)
C ROUGE-W-1.2 Average_F: 0.21557 (95%-conf.int. 0.20803 - 0.22335)
---------------------------------------------
C ROUGE-SU4 Average_R: 0.16477 (95%-conf.int. 0.15576 - 0.17383)
C ROUGE-SU4 Average_P: 0.18790 (95%-conf.int. 0.17794 - 0.19759)
C ROUGE-SU4 Average_F: 0.16377 (95%-conf.int. 0.15497 - 0.17240)
```
