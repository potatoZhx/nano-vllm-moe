# Acceptance Predictor Comprehensive Analysis

## 1. Data provenance and analysis scope

- Run directory: `/home/mumura/moe_spec/nano-vllm-moe/random_cache_srdp_scripts-1/res/run_20260614_133025`
- Tensor dataset: `/data2/group_谈海生/mumura/dynamick/predictor/random_cache_acceptance_dataset_20260614.pt`
- Checkpoint: `/home/mumura/moe_spec/nano-vllm-moe/random_cache_srdp_scripts-1/res/run_20260614_133025/best_model.pth`
- Wiki train JSONL: `/data2/group_谈海生/mumura/dynamick/predictor/random_cache_runs/wiki_random_cache_lru_ratios0.1-0.125-0.25-0.31-0.375-0.5_weights1-1-2-3-2-1_topc0.7/acceptance_summary_20260613_225251.jsonl`
- MTBench test JSONL: `/data2/group_谈海生/mumura/dynamick/predictor/random_cache_runs/mtbench_random_cache_lru_ratios0.1-0.125-0.25-0.31-0.375-0.5_weights1-1-2-3-2-1_topc0.7/acceptance_summary_20260614_125411.jsonl`
- Train tensor/JSONL label max absolute difference: 0.00000000
- Test tensor/JSONL label max absolute difference: 0.00000000
- Predictions were recomputed from `best_model.pth`; metrics are not inferred from the final JSON report alone.

## 2. Dataset composition

| Dataset | Source | Prompts | Steps | Articles | Prefill mean | Prefill p50 | Prefill min/max |
|---|---|---:|---:|---:|---:|---:|---:|
| Train | Wiki | 2000 | 30000 | 611 | 1687.1 | 1518.0 | 12/4096 |
| Test | MTBench | 200 | 3000 | 0 | 127.0 | 100.0 | 12/584 |

### Cache-ratio sampling by prompt

| Ratio | Train prompts | Test prompts |
|---:|---:|---:|
| 0.1 | 190 | 24 |
| 0.125 | 185 | 14 |
| 0.25 | 396 | 48 |
| 0.31 | 625 | 59 |
| 0.375 | 408 | 36 |
| 0.5 | 196 | 19 |

## 3. Checkpoint selection

- The trainer saves the checkpoint with minimum `val_mse`.
- Selected epoch: 20; `val_mse=0.028528`.
- Best MAE epoch: 26; best log-MAE epoch: 26.
- MSE, RMSE and R2 rank checkpoints identically on the same fixed validation set. Correlation does not measure calibration and should not replace an error metric.

## 4. Overall predictive performance

| Split | N | MAE | RMSE | R2 | Corr | Bias | Log-MAE | Const RMSE | RMSE gain % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train_full | 30000 | 0.0622 | 0.1003 | 0.8619 | 0.9285 | -0.0011 | 0.1266 | 0.2700 | 62.8362 |
| train_fit | 27000 | 0.0581 | 0.0895 | 0.8902 | 0.9440 | -0.0010 | 0.1122 | 0.2702 | 66.8700 |
| val | 3000 | 0.0994 | 0.1689 | 0.6016 | 0.7836 | -0.0018 | 0.2559 | 0.2676 | 36.8860 |
| test | 3000 | 0.1332 | 0.2336 | 0.4582 | 0.7079 | 0.0147 | 0.5300 | 0.3184 | 26.6449 |

- Train-fit to validation RMSE rises from 0.0895 to 0.1689 (88.7% increase).
- Validation to MTBench RMSE rises from 0.1689 to 0.2336 (38.3% increase).
- The model improves MTBench RMSE over a train-mean constant baseline by 26.6%, so it has real signal, but cross-domain error remains large.

## 5. Label and prediction distributions

| Split | Series | N | Mean | Std | Min | Max | P1 | P5 | P25 | P50 | P75 | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train_full | alpha_true | 30000 | 0.78587 | 0.26998 | 0.00000 | 1.00000 | 0.00861 | 0.12595 | 0.70272 | 0.90465 | 0.97663 | 0.99993 | 1.00000 |
| train_full | alpha_pred | 30000 | 0.78478 | 0.24659 | 0.00000 | 0.99993 | 0.01334 | 0.18058 | 0.71969 | 0.88689 | 0.95151 | 0.99332 | 0.99841 |
| train_fit | alpha_true | 27000 | 0.78618 | 0.27024 | 0.00000 | 1.00000 | 0.00862 | 0.12523 | 0.70397 | 0.90532 | 0.97660 | 0.99993 | 1.00000 |
| train_fit | alpha_pred | 27000 | 0.78517 | 0.24737 | 0.00000 | 0.99993 | 0.01234 | 0.17393 | 0.72285 | 0.88753 | 0.95157 | 0.99330 | 0.99841 |
| val | alpha_true | 3000 | 0.78311 | 0.26760 | 0.00003 | 1.00000 | 0.00757 | 0.13201 | 0.68814 | 0.89928 | 0.97686 | 0.99992 | 1.00000 |
| val | alpha_pred | 3000 | 0.78127 | 0.23936 | 0.00051 | 0.99958 | 0.03119 | 0.22547 | 0.69560 | 0.87912 | 0.95030 | 0.99348 | 0.99832 |
| test | alpha_true | 3000 | 0.75980 | 0.31731 | 0.00000 | 1.00000 | 0.00072 | 0.02279 | 0.63053 | 0.92408 | 0.99360 | 0.99999 | 1.00000 |
| test | alpha_pred | 3000 | 0.77445 | 0.28870 | 0.00011 | 0.99963 | 0.00629 | 0.06123 | 0.71206 | 0.91367 | 0.96765 | 0.99375 | 0.99818 |

- Train prediction standard deviation contracts from 0.2700 to 0.2466; MTBench contracts from 0.3173 to 0.2887.
- The contraction is asymmetric: low alpha is generally overestimated while near-one alpha is underestimated.

## 6. Error by true-alpha bucket

| Split | True alpha bucket | N | Share | True mean | Pred mean | MAE | RMSE | Bias | Corr | Log-MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train_full | alpha < 0.5 | 4666 | 0.1555 | 0.2343 | 0.3178 | 0.1023 | 0.1651 | 0.0835 | 0.7602 | 0.4304 |
| train_full | 0.5 <= alpha < 0.7 | 2766 | 0.0922 | 0.6116 | 0.6569 | 0.1104 | 0.1402 | 0.0453 | 0.3062 | 0.1760 |
| train_full | 0.7 <= alpha < 0.85 | 4421 | 0.1474 | 0.7864 | 0.7899 | 0.0788 | 0.1044 | 0.0035 | 0.2808 | 0.1058 |
| train_full | 0.85 <= alpha < 0.95 | 7126 | 0.2375 | 0.9071 | 0.8781 | 0.0512 | 0.0762 | -0.0290 | 0.2816 | 0.0612 |
| train_full | alpha >= 0.95 | 11021 | 0.3674 | 0.9845 | 0.9522 | 0.0336 | 0.0529 | -0.0324 | 0.4819 | 0.0361 |
| train_fit | alpha < 0.5 | 4207 | 0.1558 | 0.2339 | 0.3075 | 0.0891 | 0.1430 | 0.0735 | 0.8205 | 0.3653 |
| train_fit | 0.5 <= alpha < 0.7 | 2458 | 0.0910 | 0.6122 | 0.6597 | 0.1056 | 0.1340 | 0.0475 | 0.3255 | 0.1647 |
| train_fit | 0.7 <= alpha < 0.85 | 3950 | 0.1463 | 0.7865 | 0.7940 | 0.0745 | 0.0938 | 0.0076 | 0.3044 | 0.0965 |
| train_fit | 0.85 <= alpha < 0.95 | 6446 | 0.2387 | 0.9070 | 0.8804 | 0.0489 | 0.0667 | -0.0267 | 0.3193 | 0.0565 |
| train_fit | alpha >= 0.95 | 9939 | 0.3681 | 0.9845 | 0.9531 | 0.0325 | 0.0479 | -0.0313 | 0.5437 | 0.0345 |
| val | alpha < 0.5 | 459 | 0.1530 | 0.2373 | 0.4124 | 0.2233 | 0.2994 | 0.1750 | 0.3510 | 1.0274 |
| val | 0.5 <= alpha < 0.7 | 308 | 0.1027 | 0.6070 | 0.6349 | 0.1484 | 0.1823 | 0.0279 | 0.2021 | 0.2663 |
| val | 0.7 <= alpha < 0.85 | 471 | 0.1570 | 0.7857 | 0.7554 | 0.1144 | 0.1687 | -0.0303 | 0.2098 | 0.1841 |
| val | 0.85 <= alpha < 0.95 | 680 | 0.2267 | 0.9077 | 0.8561 | 0.0727 | 0.1369 | -0.0516 | 0.1736 | 0.1063 |
| val | alpha >= 0.95 | 1082 | 0.3607 | 0.9853 | 0.9436 | 0.0431 | 0.0863 | -0.0417 | 0.2616 | 0.0509 |
| test | alpha < 0.5 | 596 | 0.1987 | 0.1838 | 0.4074 | 0.2845 | 0.3684 | 0.2236 | 0.3337 | 1.8665 |
| test | 0.5 <= alpha < 0.7 | 262 | 0.0873 | 0.6075 | 0.6596 | 0.2162 | 0.2564 | 0.0520 | 0.1592 | 0.4638 |
| test | 0.7 <= alpha < 0.85 | 319 | 0.1063 | 0.7833 | 0.7819 | 0.1484 | 0.2191 | -0.0014 | 0.0482 | 0.3081 |
| test | 0.85 <= alpha < 0.95 | 507 | 0.1690 | 0.9074 | 0.8660 | 0.0849 | 0.1713 | -0.0413 | 0.1123 | 0.1677 |
| test | alpha >= 0.95 | 1316 | 0.4387 | 0.9884 | 0.9265 | 0.0632 | 0.1639 | -0.0620 | 0.1173 | 0.1312 |

- MTBench `alpha < 0.5` is the critical failure region: true mean 0.1838, predicted mean 0.4074, bias +0.2236, RMSE 0.3684.
- For `alpha >= 0.95`, the mean bias reverses to -0.0620; high-acceptance steps are underestimated.
- Negative within-bucket R2 is expected when each bucket has very little target variance; MAE, RMSE and bias are more useful for these rows.

## 7. Error by cache ratio

| Split | Cache ratio | N | True mean | Pred mean | MAE | RMSE | R2 | Bias | Log-MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train_full | 0.1000 | 2850 | 0.5559 | 0.5606 | 0.0853 | 0.1255 | 0.8516 | 0.0047 | 0.2482 |
| train_full | 0.1250 | 2775 | 0.5908 | 0.5912 | 0.0848 | 0.1264 | 0.8466 | 0.0005 | 0.2240 |
| train_full | 0.2500 | 5940 | 0.7480 | 0.7485 | 0.0720 | 0.1121 | 0.8370 | 0.0005 | 0.1499 |
| train_full | 0.3100 | 9375 | 0.8239 | 0.8240 | 0.0601 | 0.0971 | 0.8222 | 0.0000 | 0.1069 |
| train_full | 0.3750 | 6120 | 0.8825 | 0.8803 | 0.0483 | 0.0815 | 0.7946 | -0.0022 | 0.0735 |
| train_full | 0.5000 | 2940 | 0.9472 | 0.9344 | 0.0340 | 0.0555 | 0.6958 | -0.0127 | 0.0428 |
| train_fit | 0.1000 | 2545 | 0.5559 | 0.5590 | 0.0781 | 0.1108 | 0.8851 | 0.0031 | 0.2171 |
| train_fit | 0.1250 | 2519 | 0.5899 | 0.5918 | 0.0775 | 0.1101 | 0.8849 | 0.0019 | 0.1982 |
| train_fit | 0.2500 | 5325 | 0.7489 | 0.7496 | 0.0664 | 0.0988 | 0.8723 | 0.0007 | 0.1285 |
| train_fit | 0.3100 | 8457 | 0.8245 | 0.8245 | 0.0562 | 0.0859 | 0.8609 | 0.0001 | 0.0951 |
| train_fit | 0.3750 | 5488 | 0.8818 | 0.8798 | 0.0467 | 0.0774 | 0.8181 | -0.0020 | 0.0699 |
| train_fit | 0.5000 | 2666 | 0.9476 | 0.9350 | 0.0332 | 0.0522 | 0.7170 | -0.0126 | 0.0399 |
| val | 0.1000 | 305 | 0.5557 | 0.5738 | 0.1453 | 0.2116 | 0.5517 | 0.0181 | 0.5082 |
| val | 0.1250 | 256 | 0.5990 | 0.5855 | 0.1575 | 0.2319 | 0.4145 | -0.0135 | 0.4782 |
| val | 0.2500 | 615 | 0.7395 | 0.7384 | 0.1204 | 0.1920 | 0.5549 | -0.0011 | 0.3349 |
| val | 0.3100 | 918 | 0.8189 | 0.8185 | 0.0964 | 0.1680 | 0.4581 | -0.0004 | 0.2165 |
| val | 0.3750 | 632 | 0.8887 | 0.8843 | 0.0625 | 0.1110 | 0.5472 | -0.0044 | 0.1046 |
| val | 0.5000 | 274 | 0.9428 | 0.9288 | 0.0418 | 0.0806 | 0.5607 | -0.0139 | 0.0709 |
| test | 0.1000 | 360 | 0.3671 | 0.3837 | 0.2389 | 0.3389 | 0.0616 | 0.0166 | 1.6852 |
| test | 0.1250 | 210 | 0.4285 | 0.4745 | 0.2308 | 0.3317 | 0.1509 | 0.0460 | 1.3520 |
| test | 0.2500 | 720 | 0.7736 | 0.8082 | 0.1481 | 0.2427 | 0.2416 | 0.0346 | 0.4188 |
| test | 0.3100 | 885 | 0.8455 | 0.8493 | 0.1114 | 0.1998 | 0.2520 | 0.0038 | 0.2746 |
| test | 0.3750 | 540 | 0.8976 | 0.9052 | 0.0815 | 0.1771 | 0.1177 | 0.0076 | 0.2236 |
| test | 0.5000 | 285 | 0.9378 | 0.9237 | 0.0561 | 0.1261 | -0.1488 | -0.0141 | 0.1193 |

- At cache ratio 0.1, MTBench RMSE is 0.3389 and R2 is only 0.0616. Low-cache operation is the main generalization bottleneck.
- The ratio 0.5 row has low RMSE but negative R2 because labels are tightly concentrated near one; this is not evidence that the absolute predictions are worse than low-ratio predictions.

## 8. Error by decode step

| Split | Step | N | True mean | Pred mean | MAE | RMSE | R2 | Bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train_full | 1 | 2000 | 0.8541 | 0.8438 | 0.0527 | 0.0870 | 0.8583 | -0.0103 |
| train_full | 2 | 2000 | 0.8064 | 0.7971 | 0.0580 | 0.0945 | 0.8737 | -0.0092 |
| train_full | 3 | 2000 | 0.7920 | 0.7888 | 0.0594 | 0.0953 | 0.8794 | -0.0032 |
| train_full | 4 | 2000 | 0.7729 | 0.7718 | 0.0614 | 0.1011 | 0.8754 | -0.0011 |
| train_full | 5 | 2000 | 0.7694 | 0.7676 | 0.0631 | 0.1012 | 0.8742 | -0.0017 |
| train_full | 6 | 2000 | 0.7611 | 0.7627 | 0.0652 | 0.1074 | 0.8595 | 0.0016 |
| train_full | 7 | 2000 | 0.7593 | 0.7607 | 0.0643 | 0.1036 | 0.8650 | 0.0014 |
| train_full | 8 | 2000 | 0.7619 | 0.7677 | 0.0660 | 0.1054 | 0.8577 | 0.0058 |
| train_full | 9 | 2000 | 0.7696 | 0.7738 | 0.0651 | 0.1039 | 0.8559 | 0.0042 |
| train_full | 10 | 2000 | 0.7778 | 0.7770 | 0.0632 | 0.1012 | 0.8564 | -0.0007 |
| train_full | 11 | 2000 | 0.7828 | 0.7813 | 0.0614 | 0.0965 | 0.8715 | -0.0015 |
| train_full | 12 | 2000 | 0.7882 | 0.7882 | 0.0636 | 0.0995 | 0.8556 | -0.0001 |
| train_full | 13 | 2000 | 0.7947 | 0.7951 | 0.0627 | 0.1023 | 0.8387 | 0.0004 |
| train_full | 14 | 2000 | 0.7937 | 0.7942 | 0.0630 | 0.1011 | 0.8486 | 0.0005 |
| train_full | 15 | 2000 | 0.8044 | 0.8018 | 0.0638 | 0.1033 | 0.8309 | -0.0026 |
| train_fit | 1 | 1801 | 0.8546 | 0.8436 | 0.0502 | 0.0803 | 0.8812 | -0.0110 |
| train_fit | 2 | 1811 | 0.8036 | 0.7960 | 0.0536 | 0.0804 | 0.9105 | -0.0075 |
| train_fit | 3 | 1804 | 0.7944 | 0.7897 | 0.0545 | 0.0810 | 0.9116 | -0.0047 |
| train_fit | 4 | 1796 | 0.7726 | 0.7713 | 0.0577 | 0.0906 | 0.8993 | -0.0013 |
| train_fit | 5 | 1816 | 0.7669 | 0.7668 | 0.0590 | 0.0917 | 0.8984 | -0.0001 |
| train_fit | 6 | 1786 | 0.7611 | 0.7628 | 0.0597 | 0.0928 | 0.8957 | 0.0017 |
| train_fit | 7 | 1810 | 0.7631 | 0.7623 | 0.0600 | 0.0913 | 0.8925 | -0.0008 |
| train_fit | 8 | 1803 | 0.7603 | 0.7656 | 0.0623 | 0.0951 | 0.8848 | 0.0053 |
| train_fit | 9 | 1790 | 0.7708 | 0.7743 | 0.0595 | 0.0908 | 0.8893 | 0.0034 |
| train_fit | 10 | 1786 | 0.7787 | 0.7783 | 0.0593 | 0.0899 | 0.8856 | -0.0004 |
| train_fit | 11 | 1817 | 0.7833 | 0.7807 | 0.0573 | 0.0869 | 0.8971 | -0.0026 |
| train_fit | 12 | 1787 | 0.7887 | 0.7891 | 0.0600 | 0.0925 | 0.8748 | 0.0004 |
| train_fit | 13 | 1789 | 0.7973 | 0.7993 | 0.0588 | 0.0925 | 0.8680 | 0.0020 |
| train_fit | 14 | 1810 | 0.7929 | 0.7935 | 0.0596 | 0.0933 | 0.8727 | 0.0006 |
| train_fit | 15 | 1794 | 0.8044 | 0.8042 | 0.0594 | 0.0920 | 0.8660 | -0.0001 |
| val | 1 | 199 | 0.8497 | 0.8459 | 0.0748 | 0.1331 | 0.6112 | -0.0037 |
| val | 2 | 189 | 0.8333 | 0.8076 | 0.1003 | 0.1806 | 0.4135 | -0.0257 |
| val | 3 | 196 | 0.7703 | 0.7810 | 0.1042 | 0.1795 | 0.6192 | 0.0106 |
| val | 4 | 204 | 0.7755 | 0.7761 | 0.0943 | 0.1672 | 0.6761 | 0.0006 |
| val | 5 | 184 | 0.7936 | 0.7759 | 0.1033 | 0.1685 | 0.5830 | -0.0177 |
| val | 6 | 214 | 0.7613 | 0.7621 | 0.1111 | 0.1893 | 0.5373 | 0.0008 |
| val | 7 | 190 | 0.7228 | 0.7454 | 0.1055 | 0.1832 | 0.6521 | 0.0226 |
| val | 8 | 197 | 0.7766 | 0.7865 | 0.0996 | 0.1733 | 0.5927 | 0.0099 |
| val | 9 | 210 | 0.7591 | 0.7699 | 0.1131 | 0.1802 | 0.5827 | 0.0107 |
| val | 10 | 214 | 0.7698 | 0.7664 | 0.0959 | 0.1679 | 0.6295 | -0.0034 |
| val | 11 | 183 | 0.7775 | 0.7870 | 0.1016 | 0.1635 | 0.5723 | 0.0095 |
| val | 12 | 213 | 0.7840 | 0.7805 | 0.0939 | 0.1452 | 0.6965 | -0.0035 |
| val | 13 | 211 | 0.7722 | 0.7590 | 0.0959 | 0.1629 | 0.5858 | -0.0132 |
| val | 14 | 190 | 0.8013 | 0.8009 | 0.0957 | 0.1573 | 0.5845 | -0.0004 |
| val | 15 | 206 | 0.8043 | 0.7803 | 0.1015 | 0.1728 | 0.5203 | -0.0240 |
| test | 1 | 200 | 0.8429 | 0.8295 | 0.1279 | 0.2556 | 0.0693 | -0.0133 |
| test | 2 | 200 | 0.7801 | 0.7743 | 0.1308 | 0.2508 | 0.3636 | -0.0057 |
| test | 3 | 200 | 0.7889 | 0.7823 | 0.1300 | 0.2420 | 0.3151 | -0.0066 |
| test | 4 | 200 | 0.7796 | 0.7744 | 0.1120 | 0.1969 | 0.5862 | -0.0052 |
| test | 5 | 200 | 0.7679 | 0.7681 | 0.1195 | 0.2074 | 0.5712 | 0.0002 |
| test | 6 | 200 | 0.7605 | 0.7706 | 0.1385 | 0.2459 | 0.4169 | 0.0101 |
| test | 7 | 200 | 0.7630 | 0.7936 | 0.1262 | 0.2284 | 0.4806 | 0.0305 |
| test | 8 | 200 | 0.7637 | 0.7853 | 0.1104 | 0.1854 | 0.6343 | 0.0216 |
| test | 9 | 200 | 0.7405 | 0.7670 | 0.1350 | 0.2268 | 0.5189 | 0.0265 |
| test | 10 | 200 | 0.7332 | 0.7402 | 0.1577 | 0.2600 | 0.3533 | 0.0069 |
| test | 11 | 200 | 0.7488 | 0.7684 | 0.1387 | 0.2296 | 0.4567 | 0.0196 |
| test | 12 | 200 | 0.7202 | 0.7577 | 0.1402 | 0.2261 | 0.5615 | 0.0375 |
| test | 13 | 200 | 0.7325 | 0.7647 | 0.1367 | 0.2318 | 0.4942 | 0.0322 |
| test | 14 | 200 | 0.7465 | 0.7632 | 0.1448 | 0.2459 | 0.4248 | 0.0167 |
| test | 15 | 200 | 0.7288 | 0.7775 | 0.1505 | 0.2562 | 0.4238 | 0.0487 |

- Best MTBench step: 8 (RMSE 0.1854). Worst step: 10 (RMSE 0.2600).
- Bias becomes increasingly positive in several later steps, reaching 0.0487 at step 15.

## 9. Error by prefill length

| Split | Prefill bin | N | True mean | Pred mean | MAE | RMSE | R2 | Bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train_full | >= 4096 | 30 | 0.7086 | 0.7716 | 0.0952 | 0.1349 | 0.7738 | 0.0630 |
| train_full | [0, 64) | 495 | 0.6902 | 0.6964 | 0.0834 | 0.1224 | 0.8060 | 0.0062 |
| train_full | [1024, 2048) | 8700 | 0.7872 | 0.7868 | 0.0608 | 0.0977 | 0.8685 | -0.0004 |
| train_full | [128, 256) | 1365 | 0.8078 | 0.8065 | 0.0653 | 0.1033 | 0.8113 | -0.0013 |
| train_full | [2048, 4096) | 10740 | 0.7947 | 0.7944 | 0.0605 | 0.0993 | 0.8605 | -0.0004 |
| train_full | [256, 512) | 2865 | 0.7743 | 0.7715 | 0.0657 | 0.1065 | 0.8584 | -0.0028 |
| train_full | [512, 1024) | 5280 | 0.7735 | 0.7698 | 0.0630 | 0.1003 | 0.8699 | -0.0037 |
| train_full | [64, 128) | 525 | 0.8087 | 0.8068 | 0.0635 | 0.0977 | 0.8500 | -0.0019 |
| train_fit | >= 4096 | 28 | 0.6912 | 0.7605 | 0.1002 | 0.1394 | 0.7618 | 0.0693 |
| train_fit | [0, 64) | 442 | 0.6983 | 0.6963 | 0.0763 | 0.1079 | 0.8407 | -0.0020 |
| train_fit | [1024, 2048) | 7799 | 0.7868 | 0.7861 | 0.0566 | 0.0865 | 0.8976 | -0.0007 |
| train_fit | [128, 256) | 1232 | 0.8064 | 0.8068 | 0.0622 | 0.0949 | 0.8409 | 0.0004 |
| train_fit | [2048, 4096) | 9699 | 0.7956 | 0.7952 | 0.0563 | 0.0879 | 0.8908 | -0.0004 |
| train_fit | [256, 512) | 2570 | 0.7763 | 0.7730 | 0.0608 | 0.0945 | 0.8882 | -0.0033 |
| train_fit | [512, 1024) | 4743 | 0.7721 | 0.7699 | 0.0594 | 0.0914 | 0.8929 | -0.0022 |
| train_fit | [64, 128) | 487 | 0.8119 | 0.8098 | 0.0588 | 0.0884 | 0.8732 | -0.0021 |
| val | >= 4096 | 2 | 0.9526 | 0.9273 | 0.0253 | 0.0262 | -1.5386 | -0.0253 |
| val | [0, 64) | 53 | 0.6226 | 0.6972 | 0.1428 | 0.2068 | 0.5971 | 0.0745 |
| val | [1024, 2048) | 901 | 0.7902 | 0.7926 | 0.0973 | 0.1652 | 0.5964 | 0.0024 |
| val | [128, 256) | 133 | 0.8208 | 0.8040 | 0.0933 | 0.1611 | 0.5298 | -0.0167 |
| val | [2048, 4096) | 1041 | 0.7869 | 0.7863 | 0.0992 | 0.1725 | 0.5755 | -0.0006 |
| val | [256, 512) | 295 | 0.7564 | 0.7583 | 0.1076 | 0.1803 | 0.6067 | 0.0019 |
| val | [512, 1024) | 537 | 0.7855 | 0.7681 | 0.0944 | 0.1585 | 0.6482 | -0.0174 |
| val | [64, 128) | 38 | 0.7670 | 0.7680 | 0.1236 | 0.1778 | 0.6356 | 0.0010 |
| test | [0, 64) | 960 | 0.7323 | 0.7456 | 0.1394 | 0.2369 | 0.4463 | 0.0133 |
| test | [128, 256) | 870 | 0.7666 | 0.7889 | 0.1372 | 0.2460 | 0.4370 | 0.0223 |
| test | [256, 512) | 285 | 0.7364 | 0.7603 | 0.1552 | 0.2598 | 0.3760 | 0.0239 |
| test | [512, 1024) | 30 | 0.9087 | 0.9121 | 0.0605 | 0.0927 | 0.5326 | 0.0034 |
| test | [64, 128) | 855 | 0.7864 | 0.7921 | 0.1175 | 0.2098 | 0.5154 | 0.0057 |

- Wiki prefill median is 1518 tokens, while MTBench median is 100. Most MTBench samples occupy a region weakly represented by the training distribution.

## 10. Threshold decision risk

`False-safe rate` is the fraction of truly below-threshold examples incorrectly predicted above the threshold. This is the dangerous error when predictions control aggressive drafting.

| Split | Threshold | Accuracy | Precision | Recall | Specificity | False-safe | False-conservative | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train_full | 0.5000 | 0.9535 | 0.9626 | 0.9831 | 0.7928 | 0.2072 | 0.0169 | 967 | 428 |
| train_full | 0.7000 | 0.9261 | 0.9437 | 0.9590 | 0.8262 | 0.1738 | 0.0410 | 1292 | 926 |
| train_full | 0.8500 | 0.8674 | 0.9029 | 0.8749 | 0.8559 | 0.1441 | 0.1251 | 1708 | 2271 |
| train_full | 0.9500 | 0.8353 | 0.8952 | 0.6249 | 0.9575 | 0.0425 | 0.3751 | 806 | 4134 |
| train_fit | 0.5000 | 0.9596 | 0.9660 | 0.9869 | 0.8117 | 0.1883 | 0.0131 | 792 | 299 |
| train_fit | 0.7000 | 0.9319 | 0.9469 | 0.9636 | 0.8353 | 0.1647 | 0.0364 | 1098 | 741 |
| train_fit | 0.8500 | 0.8703 | 0.9054 | 0.8781 | 0.8583 | 0.1417 | 0.1219 | 1504 | 1998 |
| train_fit | 0.9500 | 0.8349 | 0.8948 | 0.6248 | 0.9572 | 0.0428 | 0.3752 | 730 | 3729 |
| val | 0.5000 | 0.8987 | 0.9324 | 0.9492 | 0.6187 | 0.3813 | 0.0508 | 175 | 129 |
| val | 0.7000 | 0.8737 | 0.9135 | 0.9172 | 0.7471 | 0.2529 | 0.0828 | 194 | 185 |
| val | 0.8500 | 0.8410 | 0.8795 | 0.8451 | 0.8352 | 0.1648 | 0.1549 | 204 | 273 |
| val | 0.9500 | 0.8397 | 0.8991 | 0.6257 | 0.9604 | 0.0396 | 0.3743 | 76 | 405 |
| test | 0.5000 | 0.8730 | 0.9087 | 0.9355 | 0.6208 | 0.3792 | 0.0645 | 226 | 155 |
| test | 0.7000 | 0.8557 | 0.8769 | 0.9281 | 0.6748 | 0.3252 | 0.0719 | 279 | 154 |
| test | 0.8500 | 0.8227 | 0.8406 | 0.8738 | 0.7434 | 0.2566 | 0.1262 | 302 | 230 |
| test | 0.9500 | 0.8003 | 0.8354 | 0.6786 | 0.8955 | 0.1045 | 0.3214 | 176 | 423 |

- On MTBench, false-safe rates are 37.9%, 32.5%, 25.7% and 10.5% for thresholds 0.5, 0.7, 0.85 and 0.95 respectively.
- Therefore raw predictions should not yet be used as an unsafe go/no-go control signal without calibration or a conservative lower bound.

## 11. Calibration

| Split | Weighted abs calibration error | Max bin error |
|---|---:|---:|
| train_full | 0.0053 | 0.0203 |
| train_fit | 0.0075 | 0.0323 |
| val | 0.0230 | 0.1721 |
| test | 0.0646 | 0.2709 |

- MTBench weighted calibration error is materially worse than Wiki train and validation, confirming that domain shift affects probability calibration rather than only ranking.

## 12. Prompt-chain performance

| Split | Prompts | True expected accepted | Pred expected accepted | MAE steps | RMSE steps | Bias steps | R2 | Corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train_full | 2000 | 5.6605 | 5.3702 | 0.7712 | 1.1169 | -0.2903 | 0.9109 | 0.9605 |
| test | 200 | 5.6679 | 5.5143 | 1.7216 | 2.5498 | -0.1536 | 0.6097 | 0.7936 |

- MTBench mean chain bias is small (-0.1536 steps), but per-prompt MAE is 1.7216 steps and RMSE is 2.5498 steps. Mean cancellation hides large individual prompt errors.

## 13. Validation leakage and Wiki window overlap

- Train/validation splitting is performed at step level.
- Validation prompt overlap with train: 1573/1573 (100.0%).
- Validation article overlap with train: 587/587 (100.0%).
- Same-article window pairs: 3285; overlapping pairs: 1815 (55.3%).
- Mean overlap fraction among overlapping pairs: 77.4%.
- MTBench test remains an independent external test set, so its reported metrics are not invalidated by the Wiki train/validation leakage. The leakage does make validation-based checkpoint selection optimistic.

## 14. Train-to-test distribution drift

| Feature | Train mean | Test mean | Mean shift | SMD | KS | PSI |
|---|---:|---:|---:|---:|---:|---:|
| alpha_true | 0.7859 | 0.7598 | -0.0261 | -0.0885 | 0.0932 | 0.0944 |
| alpha_pred | 0.7848 | 0.7745 | -0.0103 | -0.0385 | 0.1005 | 0.0869 |
| cache_ratio | 0.2929 | 0.2872 | -0.0057 | -0.0519 | 0.0445 | 0.0237 |
| prefill_len | 1687.1060 | 127.0400 | -1560.0660 | -1.9600 | 0.8555 | 11.1409 |

- Prefill length is the dominant explicit shift: SMD -1.9600, KS 0.8555, PSI 11.1409.
- Across all 2882 feature dimensions, 221 have `|SMD| >= 0.5`.

### Feature branch summary

| Branch | Dims | Train mean | Train std | Test mean | Test std | Mean shift |
|---|---:|---:|---:|---:|---:|---:|
| hidden | 2048 | -0.0151 | 3.1657 | -0.0039 | 3.0831 | 0.0111 |
| history | 11 | 0.6419 | 0.7162 | 0.5107 | 0.5664 | -0.1312 |
| route_raw | 768 | 0.1250 | 0.0657 | 0.1250 | 0.0663 | 0.0000 |
| route_summary | 45 | 0.5130 | 0.6805 | 0.5086 | 0.6787 | -0.0044 |
| token_features | 10 | 11.7190 | 11.9306 | 12.5546 | 12.8162 | 0.8356 |

### Top 20 shifted feature dimensions

| Branch | Dimension | Train mean | Train std | Test mean | Test std | SMD |
|---|---:|---:|---:|---:|---:|---:|
| history | 10 | 0.2084 | 0.1385 | 0.0157 | 0.0126 | -1.9600 |
| hidden | 1738 | -4.1621 | 4.1315 | 4.2184 | 5.4039 | 1.7423 |
| hidden | 857 | 3.4745 | 4.0786 | -1.9197 | 4.4132 | -1.2695 |
| hidden | 1003 | -6.9786 | 4.5016 | -0.8903 | 5.5284 | 1.2077 |
| hidden | 912 | 10.7911 | 8.3118 | 0.0711 | 10.2209 | -1.1508 |
| hidden | 1786 | -3.8352 | 3.1549 | 0.5077 | 4.3345 | 1.1456 |
| hidden | 707 | 11.0537 | 6.0259 | 2.8812 | 8.7784 | -1.0855 |
| hidden | 1630 | -0.8557 | 1.2383 | 0.4201 | 1.2950 | 1.0070 |
| hidden | 799 | 0.8835 | 1.2735 | -0.4429 | 1.4665 | -0.9658 |
| hidden | 1187 | -0.7545 | 2.4297 | 1.7131 | 2.7173 | 0.9574 |
| hidden | 610 | -1.9435 | 1.2099 | -0.7536 | 1.2883 | 0.9521 |
| hidden | 761 | -0.1845 | 1.2485 | -1.3971 | 1.3244 | -0.9422 |
| hidden | 1543 | -2.7999 | 1.6071 | -1.3466 | 1.5171 | 0.9300 |
| hidden | 1188 | -0.8441 | 1.2617 | 0.3662 | 1.4319 | 0.8969 |
| hidden | 449 | 0.6977 | 1.3145 | -0.5896 | 1.5779 | -0.8865 |
| hidden | 46 | 0.2100 | 1.1987 | -0.8690 | 1.3374 | -0.8496 |
| hidden | 1057 | -0.6228 | 1.8021 | 1.0612 | 2.1631 | 0.8459 |
| hidden | 361 | 0.8812 | 1.2820 | -0.2444 | 1.3784 | -0.8456 |
| hidden | 1091 | 1.2395 | 2.2435 | -0.7503 | 2.4967 | -0.8383 |
| hidden | 1293 | -0.3241 | 1.3334 | 0.8929 | 1.5695 | 0.8357 |

## 15. Overall usability assessment

**Current status: useful as a ranking/offline diagnostic model, not yet safe as a direct online drafting controller.**

Evidence supporting usefulness:

- MTBench correlation is 0.7079 and R2 is 0.4582.
- It reduces RMSE by 26.6% relative to a constant train-mean baseline.
- Cache-ratio and step trends are captured well enough for aggregate analysis.

Evidence preventing direct aggressive deployment:

- Low-alpha RMSE is 0.3684 with +0.2236 bias.
- MTBench calibration error is 0.0646.
- Prompt-chain MAE is 1.7216 accepted steps.
- Validation is contaminated by adjacent steps and same-article windows.
- The train/test prefill and hidden-state distributions differ materially.

## 16. Predictor optimization plan

### Priority 0: repair evaluation before comparing new models

1. Split Wiki by `article_id`, not by step. All windows and decode steps from one article must remain in one split.
2. Keep MTBench as untouched external test. Add a second conversation-style validation set for checkpoint selection and calibration.
3. Save metadata (`req_id`, `article_id`, `decode_step`, `cache_ratio`, `prefill_len`) into the `.pt` dataset so grouped evaluation does not depend on positional reconstruction.
4. Stop silently swallowing dataset parsing exceptions; count and report rejected records with reasons.

### Priority 1: fix training-data coverage

1. Add short-prefill Wiki samples and conversation/instruction data. Match the intended deployment prefill distribution instead of training mostly around 1K-4K tokens.
2. Oversample `alpha < 0.5`, especially very small alpha. Train P5 is substantially higher than MTBench P5, so the most dangerous tail is underrepresented.
3. Increase low cache-ratio coverage or use balanced batches over `cache_ratio x alpha_bucket x prefill_bin x decode_step`.
4. Prevent heavily sampled long Wiki articles from dominating by weighting articles or sampling an equal number of windows per article per epoch.

### Priority 2: change objective and checkpoint selection

1. Use a weighted loss that upweights low-alpha examples and false-safe overestimation. A practical form is asymmetric Huber/MSE with larger weight when `pred > target` and `target < 0.5`.
2. Retain log-domain loss for chain sensitivity, but tune its weight on a clean grouped validation set.
3. Select checkpoints using a deployment score such as: grouped validation RMSE + low-alpha MAE + threshold false-safe penalty + chain MAE.
4. Report confidence intervals across prompts/articles, not only per-step point estimates.

### Priority 3: improve normalization and representation

1. Compute per-feature train mean/std and apply fixed standardization at train and inference time. Current branch LayerNorm mixes heterogeneous features and makes a large prefill scalar affect the normalization of the whole history vector.
2. Normalize prefill length with a stable transform such as `log1p(prefill_len)` followed by train-set standardization, rather than a single `/8096` scale.
3. Reduce hidden-state domain dependence: compare hidden ablation, PCA/random projection, stronger bottleneck, dropout and domain-adversarial regularization.
4. Add explicit categorical/continuous embeddings for cache ratio and decode step instead of relying only on indirect route/history signals.

### Priority 4: add information that is available online

1. Add q-distribution confidence features: full-vocabulary entropy or log-sum-exp statistics, top-k probability mass, tail mass, top-1/top-2 margins and effective vocabulary size.
2. Add router identity features, not only router weights: original/modified expert overlap, replacement count, weighted replaced mass, rank changes and early/middle/late layer summaries.
3. Add cache-state features per layer and aggregate: hit/miss count, cached expert frequency, candidate-pool size and replacement randomness.
4. Run branch ablations (`route`, `token`, `hidden`, `history`) to determine which features genuinely transfer from Wiki to MTBench.

### Priority 5: make online decisions conservative

1. Calibrate on clean conversation validation data using isotonic regression or a small monotonic calibrator.
2. Predict uncertainty using an ensemble, quantile regression or conformal residual bounds. Drive drafting with a lower confidence bound, not the mean.
3. Optimize the actual policy metric: latency/speedup subject to a maximum false-safe or rollback-rate constraint.
4. Start deployment in shadow mode, logging predicted alpha, realized acceptance and policy decisions by cache ratio and step.

## 17. Recommended next experiment sequence

1. Rebuild article-grouped train/validation splits and reproduce the current architecture as the trustworthy baseline.
2. Retrain with balanced short-prefill/conversation data and low-alpha oversampling; do not change architecture yet.
3. Add fixed per-feature normalization and explicit cache-ratio/step inputs.
4. Compare asymmetric low-alpha loss against the existing MSE + log-MSE loss.
5. Run branch ablations, then modify architecture only for branches that show transferable value.
6. Calibrate the best model and evaluate false-safe rate plus chain MAE before any online control experiment.
