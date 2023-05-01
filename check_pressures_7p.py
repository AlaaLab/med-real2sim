import matplotlib.pylab as plt
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import os
from skimage.transform import rescale, resize
import torch.nn.functional as F
from torch.utils.data import Subset

import time
from scipy.integrate import odeint #collection of advanced numerical algorithms to solve initial-value problems of ordinary differential equations.
from matplotlib import pyplot as plt
import random
import sys

# the text as a string
texts = []
#texts.append("tensor(147.6000) tensor(68.7000) tensor(53.4553)")
texts.append("tensor(95.) tensor(40.9000) tensor(56.9474) tensor(73.2000) tensor(28.9000) tensor(60.5191) tensor(45.7000) tensor(17.9000) tensor(60.8315) tensor(85.9000) tensor(36.7000) tensor(57.2759) tensor(70.5000) tensor(32.8000) tensor(53.4752) tensor(61.8000) tensor(24.5000) tensor(60.3560) tensor(71.) tensor(21.5000) tensor(69.7183) tensor(159.4000) tensor(86.) tensor(46.0477) tensor(135.5000) tensor(61.8000) tensor(54.3911) tensor(48.8000) tensor(14.4000) tensor(70.4918)")
texts.append("tensor(50.6000) tensor(26.1000) tensor(48.4190) tensor(103.9000) tensor(47.7000) tensor(54.0905) tensor(156.5000) tensor(93.1000) tensor(40.5112) tensor(93.7000) tensor(31.6000) tensor(66.2753) tensor(143.7000) tensor(88.5000) tensor(38.4134) tensor(94.9000) tensor(55.8000) tensor(41.2013) tensor(216.8000) tensor(135.8000) tensor(37.3616) tensor(85.) tensor(33.7000) tensor(60.3529) tensor(44.2000) tensor(18.8000) tensor(57.4661) tensor(48.8000) tensor(29.2000) tensor(40.1639)")
texts.append("tensor(88.3000) tensor(36.4000) tensor(58.7769) tensor(52.2000) tensor(26.4000) tensor(49.4253) tensor(91.1000) tensor(51.3000) tensor(43.6883) tensor(124.7000) tensor(62.1000) tensor(50.2005) tensor(95.) tensor(40.7000) tensor(57.1579) tensor(150.1000) tensor(109.9000) tensor(26.7821) tensor(70.7000) tensor(26.6000) tensor(62.3762) tensor(124.1000) tensor(64.9000) tensor(47.7035) tensor(79.3000) tensor(34.2000) tensor(56.8726) tensor(56.7000) tensor(27.8000) tensor(50.9700)")
texts.append("tensor(42.1000) tensor(17.3000) tensor(58.9074) tensor(89.3000) tensor(57.9000) tensor(35.1624) tensor(148.3000) tensor(91.7000) tensor(38.1659) tensor(142.1000) tensor(105.9000) tensor(25.4750) tensor(76.3000) tensor(49.8000) tensor(34.7313) tensor(65.3000) tensor(18.) tensor(72.4349) tensor(91.6000) tensor(39.1000) tensor(57.3144) tensor(79.9000) tensor(42.) tensor(47.4343) tensor(116.2000) tensor(57.) tensor(50.9466) tensor(67.5000) tensor(25.9000) tensor(61.6296)")
texts.append("tensor(78.4000) tensor(25.5000) tensor(67.4745) tensor(80.3000) tensor(34.8000) tensor(56.6625) tensor(92.) tensor(36.7000) tensor(60.1087) tensor(122.6000) tensor(46.5000) tensor(62.0718) tensor(76.3000) tensor(38.6000) tensor(49.4102) tensor(81.4000) tensor(44.) tensor(45.9459) tensor(70.9000) tensor(30.1000) tensor(57.5458) tensor(71.3000) tensor(12.9000) tensor(81.9074) tensor(55.3000) tensor(30.1000) tensor(45.5696) tensor(100.1000) tensor(61.4000) tensor(38.6613)")
texts.append("tensor(89.4000) tensor(59.9000) tensor(32.9978) tensor(126.8000) tensor(89.2000) tensor(29.6530) tensor(90.7000) tensor(26.3000) tensor(71.0033) tensor(51.7000) tensor(19.5000) tensor(62.2824) tensor(136.8000) tensor(69.1000) tensor(49.4883) tensor(63.) tensor(9.1000) tensor(85.5556) tensor(89.7000) tensor(38.1000) tensor(57.5251) tensor(121.9000) tensor(81.8000) tensor(32.8958) tensor(56.5000) tensor(17.3000) tensor(69.3805) tensor(82.7000) tensor(42.5000) tensor(48.6094)")
texts.append("tensor(123.6000) tensor(83.1000) tensor(32.7670) tensor(94.6000) tensor(50.2000) tensor(46.9345) tensor(91.7000) tensor(42.9000) tensor(53.2170) tensor(119.1000) tensor(56.4000) tensor(52.6448) tensor(140.2000) tensor(86.5000) tensor(38.3024) tensor(111.3000) tensor(41.4000) tensor(62.8032) tensor(68.9000) tensor(33.) tensor(52.1045) tensor(73.5000) tensor(25.1000) tensor(65.8503) tensor(64.7000) tensor(38.5000) tensor(40.4946) tensor(62.7000) tensor(26.8000) tensor(57.2568)")
texts.append("tensor(117.4000) tensor(56.3000) tensor(52.0443) tensor(95.8000) tensor(59.7000) tensor(37.6827) tensor(111.6000) tensor(52.9000) tensor(52.5986) tensor(86.) tensor(57.5000) tensor(33.1395) tensor(73.1000) tensor(38.2000) tensor(47.7428) tensor(83.5000) tensor(52.6000) tensor(37.0060) tensor(95.9000) tensor(39.8000) tensor(58.4984) tensor(123.8000) tensor(56.2000) tensor(54.6042) tensor(110.5000) tensor(71.3000) tensor(35.4751) tensor(74.4000) tensor(43.9000) tensor(40.9946)")
texts.append("tensor(100.3000) tensor(52.8000) tensor(47.3579) tensor(109.6000) tensor(65.8000) tensor(39.9635) tensor(154.8000) tensor(125.6000) tensor(18.8631) tensor(80.9000) tensor(42.9000) tensor(46.9716) tensor(133.7000) tensor(92.4000) tensor(30.8901) tensor(48.4000) tensor(21.3000) tensor(55.9917) tensor(138.7000) tensor(55.7000) tensor(59.8414) tensor(73.2000) tensor(35.7000) tensor(51.2295) tensor(60.6000) tensor(34.8000) tensor(42.5743) tensor(103.1000) tensor(37.6000) tensor(63.5306)")
texts.append("tensor(66.6000) tensor(23.3000) tensor(65.0150) tensor(126.3000) tensor(51.7000) tensor(59.0657) tensor(56.) tensor(25.4000) tensor(54.6429) tensor(87.4000) tensor(36.7000) tensor(58.0092) tensor(106.9000) tensor(63.5000) tensor(40.5987) tensor(89.2000) tensor(44.6000) tensor(50.) tensor(78.) tensor(31.8000) tensor(59.2308) tensor(58.) tensor(31.3000) tensor(46.0345) tensor(116.2000) tensor(48.) tensor(58.6919) tensor(57.9000) tensor(12.9000) tensor(77.7202)")
texts.append("tensor(152.5000) tensor(95.4000) tensor(37.4426) tensor(75.8000) tensor(18.8000) tensor(75.1979) tensor(198.8000) tensor(136.) tensor(31.5895) tensor(99.7000) tensor(51.5000) tensor(48.3450) tensor(157.6000) tensor(77.4000) tensor(50.8883) tensor(67.8000) tensor(29.4000) tensor(56.6372) tensor(61.2000) tensor(33.) tensor(46.0784) tensor(128.6000) tensor(75.1000) tensor(41.6019) tensor(91.7000) tensor(33.5000) tensor(63.4678) tensor(80.1000) tensor(25.3000) tensor(68.4145)")
texts.append("tensor(92.3000) tensor(39.7000) tensor(56.9881) tensor(110.9000) tensor(53.9000) tensor(51.3977) tensor(154.8000) tensor(80.4000) tensor(48.0620) tensor(57.2000) tensor(21.8000) tensor(61.8881) tensor(88.2000) tensor(50.4000) tensor(42.8571) tensor(74.5000) tensor(31.5000) tensor(57.7181) tensor(87.6000) tensor(42.3000) tensor(51.7123) tensor(335.6000) tensor(315.5000) tensor(5.9893) tensor(127.3000) tensor(72.5000) tensor(43.0479) tensor(90.5000) tensor(25.2000) tensor(72.1547)")
texts.append("tensor(138.2000) tensor(62.8000) tensor(54.5586) tensor(69.8000) tensor(23.8000) tensor(65.9026) tensor(127.8000) tensor(67.8000) tensor(46.9484) tensor(137.) tensor(77.4000) tensor(43.5036) tensor(40.9000) tensor(20.9000) tensor(48.8998) tensor(165.7000) tensor(98.1000) tensor(40.7966) tensor(64.1000) tensor(42.4000) tensor(33.8533) tensor(51.9000) tensor(22.2000) tensor(57.2254) tensor(110.9000) tensor(75.) tensor(32.3715) tensor(109.4000) tensor(56.1000) tensor(48.7203)")
texts.append("tensor(83.2000) tensor(51.2000) tensor(38.4615) tensor(80.7000) tensor(40.6000) tensor(49.6902) tensor(125.7000) tensor(50.) tensor(60.2228) tensor(120.) tensor(72.7000) tensor(39.4167) tensor(78.8000) tensor(38.1000) tensor(51.6497) tensor(113.1000) tensor(36.6000) tensor(67.6393) tensor(77.6000) tensor(30.) tensor(61.3402) tensor(82.6000) tensor(36.5000) tensor(55.8111) tensor(68.4000) tensor(33.6000) tensor(50.8772) tensor(40.3000) tensor(17.) tensor(57.8164)")
texts.append("tensor(181.9000) tensor(115.2000) tensor(36.6685) tensor(53.7000) tensor(27.1000) tensor(49.5345) tensor(53.7000) tensor(18.9000) tensor(64.8045) tensor(90.4000) tensor(46.1000) tensor(49.0044) tensor(78.5000) tensor(27.9000) tensor(64.4586) tensor(86.1000) tensor(32.4000) tensor(62.3693) tensor(62.1000) tensor(24.5000) tensor(60.5475) tensor(76.8000) tensor(26.) tensor(66.1458) tensor(323.7000) tensor(196.4000) tensor(39.3265) tensor(218.6000) tensor(139.1000) tensor(36.3678)")
texts.append("tensor(78.6000) tensor(25.1000) tensor(68.0662) tensor(80.5000) tensor(31.) tensor(61.4907) tensor(79.9000) tensor(26.5000) tensor(66.8335) tensor(85.3000) tensor(33.2000) tensor(61.0785) tensor(54.1000) tensor(18.9000) tensor(65.0647) tensor(70.4000) tensor(18.6000) tensor(73.5796) tensor(119.9000) tensor(46.7000) tensor(61.0509) tensor(80.6000) tensor(35.6000) tensor(55.8313) tensor(135.6000) tensor(98.7000) tensor(27.2124) tensor(128.2000) tensor(63.8000) tensor(50.2340)")
texts.append("tensor(63.1000) tensor(18.1000) tensor(71.3154) tensor(104.7000) tensor(51.2000) tensor(51.0984) tensor(57.) tensor(18.3000) tensor(67.8947) tensor(96.2000) tensor(61.9000) tensor(35.6549) tensor(98.1000) tensor(54.3000) tensor(44.6483) tensor(65.7000) tensor(19.3000) tensor(70.6240) tensor(105.1000) tensor(36.6000) tensor(65.1760) tensor(109.7000) tensor(60.6000) tensor(44.7584) tensor(54.2000) tensor(16.5000) tensor(69.5572) tensor(73.7000) tensor(30.2000) tensor(59.0231)")
texts.append("tensor(70.8000) tensor(37.) tensor(47.7401) tensor(92.5000) tensor(41.6000) tensor(55.0270) tensor(121.6000) tensor(50.9000) tensor(58.1414) tensor(110.4000) tensor(52.2000) tensor(52.7174) tensor(65.3000) tensor(28.8000) tensor(55.8959) tensor(142.6000) tensor(55.3000) tensor(61.2202) tensor(112.2000) tensor(66.2000) tensor(40.9982) tensor(141.8000) tensor(55.9000) tensor(60.5783) tensor(67.7000) tensor(46.5000) tensor(31.3146) tensor(60.9000) tensor(23.9000) tensor(60.7553)")
texts.append("tensor(99.7000) tensor(46.4000) tensor(53.4604) tensor(72.4000) tensor(49.2000) tensor(32.0442) tensor(102.2000) tensor(53.6000) tensor(47.5538) tensor(37.9000) tensor(28.9000) tensor(23.7467) tensor(73.5000) tensor(44.9000) tensor(38.9116) tensor(109.8000) tensor(60.9000) tensor(44.5355) tensor(70.8000) tensor(32.) tensor(54.8023) tensor(117.3000) tensor(55.7000) tensor(52.5149) tensor(51.9000) tensor(16.3000) tensor(68.5934) tensor(141.7000) tensor(46.9000) tensor(66.9019)")
texts.append("tensor(194.1000) tensor(104.3000) tensor(46.2648) tensor(188.1000) tensor(107.4000) tensor(42.9027) tensor(47.5000) tensor(12.7000) tensor(73.2632) tensor(65.5000) tensor(24.2000) tensor(63.0534) tensor(63.2000) tensor(34.5000) tensor(45.4114) tensor(64.7000) tensor(26.1000) tensor(59.6600) tensor(101.6000) tensor(46.7000) tensor(54.0354) tensor(116.7000) tensor(81.1000) tensor(30.5056) tensor(173.6000) tensor(118.5000) tensor(31.7396) tensor(126.1000) tensor(76.5000) tensor(39.3339)")
texts.append("tensor(73.4000) tensor(33.1000) tensor(54.9046) tensor(76.7000) tensor(26.4000) tensor(65.5802) tensor(147.9000) tensor(88.4000) tensor(40.2299) tensor(82.4000) tensor(32.8000) tensor(60.1942) tensor(58.1000) tensor(25.6000) tensor(55.9380) tensor(86.2000) tensor(34.6000) tensor(59.8608) tensor(130.3000) tensor(107.7000) tensor(17.3446) tensor(112.7000) tensor(36.9000) tensor(67.2582) tensor(130.2000) tensor(68.8000) tensor(47.1582) tensor(103.4000) tensor(35.6000) tensor(65.5706)")
texts.append("tensor(115.) tensor(62.1000) tensor(46.) tensor(92.3000) tensor(50.1000) tensor(45.7205) tensor(54.9000) tensor(23.) tensor(58.1056) tensor(113.4000) tensor(46.8000) tensor(58.7302) tensor(73.2000) tensor(46.2000) tensor(36.8852) tensor(94.4000) tensor(55.5000) tensor(41.2076) tensor(76.6000) tensor(33.6000) tensor(56.1358) tensor(83.2000) tensor(40.9000) tensor(50.8413) tensor(113.5000) tensor(51.4000) tensor(54.7137) tensor(76.5000) tensor(28.8000) tensor(62.3529)")
texts.append("tensor(109.3000) tensor(47.6000) tensor(56.4501) tensor(195.5000) tensor(104.9000) tensor(46.3427) tensor(97.7000) tensor(45.) tensor(53.9406) tensor(44.3000) tensor(21.3000) tensor(51.9187) tensor(71.2000) tensor(24.8000) tensor(65.1685) tensor(118.) tensor(43.9000) tensor(62.7966) tensor(97.9000) tensor(37.1000) tensor(62.1042) tensor(132.1000) tensor(69.7000) tensor(47.2369) tensor(116.4000) tensor(36.) tensor(69.0722) tensor(64.1000) tensor(20.1000) tensor(68.6427)")
texts.append("tensor(106.4000) tensor(32.7000) tensor(69.2669) tensor(88.1000) tensor(26.6000) tensor(69.8070) tensor(65.2000) tensor(28.3000) tensor(56.5951) tensor(89.1000) tensor(50.4000) tensor(43.4343) tensor(78.7000) tensor(34.3000) tensor(56.4168) tensor(91.3000) tensor(29.7000) tensor(67.4699) tensor(71.5000) tensor(31.9000) tensor(55.3846) tensor(92.) tensor(50.5000) tensor(45.1087) tensor(108.5000) tensor(55.5000) tensor(48.8479) tensor(61.3000) tensor(25.7000) tensor(58.0750)")
#240:
texts.append("tensor(102.3000) tensor(50.5000) tensor(50.6354) tensor(96.8000) tensor(56.1000) tensor(42.0455) tensor(55.) tensor(26.5000) tensor(51.8182) tensor(61.1000) tensor(23.7000) tensor(61.2111) tensor(76.4000) tensor(34.6000) tensor(54.7120) tensor(38.) tensor(21.5000) tensor(43.4211) tensor(132.4000) tensor(84.4000) tensor(36.2538) tensor(244.2000) tensor(162.3000) tensor(33.5381) tensor(103.9000) tensor(44.4000) tensor(57.2666) tensor(78.6000) tensor(39.9000) tensor(49.2366)")
texts.append("tensor(77.8000) tensor(36.2000) tensor(53.4704) tensor(65.6000) tensor(32.3000) tensor(50.7622) tensor(71.2000) tensor(36.1000) tensor(49.2978) tensor(131.5000) tensor(83.9000) tensor(36.1977) tensor(77.7000) tensor(36.6000) tensor(52.8958) tensor(109.2000) tensor(70.2000) tensor(35.7143) tensor(107.9000) tensor(65.1000) tensor(39.6664) tensor(64.8000) tensor(22.8000) tensor(64.8148) tensor(110.8000) tensor(61.7000) tensor(44.3141) tensor(153.3000) tensor(60.4000) tensor(60.6001)")
texts.append("tensor(69.1000) tensor(28.4000) tensor(58.9001) tensor(75.3000) tensor(33.7000) tensor(55.2457) tensor(103.5000) tensor(51.2000) tensor(50.5314) tensor(78.7000) tensor(32.8000) tensor(58.3227) tensor(57.3000) tensor(17.9000) tensor(68.7609) tensor(127.5000) tensor(65.4000) tensor(48.7059) tensor(112.7000) tensor(79.6000) tensor(29.3700) tensor(78.8000) tensor(56.6000) tensor(28.1726) tensor(101.3000) tensor(41.) tensor(59.5262) tensor(138.4000) tensor(72.4000) tensor(47.6879)")
texts.append("tensor(71.9000) tensor(21.1000) tensor(70.6537) tensor(110.1000) tensor(60.9000) tensor(44.6866) tensor(84.5000) tensor(35.9000) tensor(57.5148) tensor(86.4000) tensor(40.3000) tensor(53.3565) tensor(67.9000) tensor(23.6000) tensor(65.2430) tensor(74.2000) tensor(28.6000) tensor(61.4555) tensor(137.9000) tensor(85.8000) tensor(37.7810) tensor(93.1000) tensor(41.8000) tensor(55.1020) tensor(94.) tensor(66.1000) tensor(29.6809) tensor(94.6000) tensor(44.4000) tensor(53.0655)")
texts.append("tensor(111.3000) tensor(103.5000) tensor(7.0081) tensor(54.3000) tensor(21.) tensor(61.3260) tensor(100.3000) tensor(36.7000) tensor(63.4098) tensor(89.6000) tensor(55.6000) tensor(37.9464) tensor(97.6000) tensor(37.5000) tensor(61.5779) tensor(113.1000) tensor(43.3000) tensor(61.7153) tensor(80.3000) tensor(26.6000) tensor(66.8742) tensor(86.4000) tensor(36.9000) tensor(57.2917) tensor(150.2000) tensor(102.8000) tensor(31.5579) tensor(141.4000) tensor(84.3000) tensor(40.3819)")
texts.append("tensor(94.2000) tensor(50.) tensor(46.9214) tensor(101.4000) tensor(53.) tensor(47.7318) tensor(109.6000) tensor(43.8000) tensor(60.0365) tensor(151.8000) tensor(77.9000) tensor(48.6825) tensor(67.4000) tensor(31.9000) tensor(52.6706) tensor(94.) tensor(34.6000) tensor(63.1915) tensor(82.5000) tensor(30.5000) tensor(63.0303) tensor(88.4000) tensor(43.5000) tensor(50.7919) tensor(93.8000) tensor(43.2000) tensor(53.9446) tensor(69.2000) tensor(39.7000) tensor(42.6301)")
texts.append("tensor(216.9000) tensor(124.) tensor(42.8308) tensor(75.9000) tensor(31.9000) tensor(57.9710) tensor(63.8000) tensor(34.) tensor(46.7085) tensor(134.9000) tensor(73.6000) tensor(45.4411) tensor(62.3000) tensor(20.3000) tensor(67.4157) tensor(121.7000) tensor(54.4000) tensor(55.2999) tensor(94.8000) tensor(55.2000) tensor(41.7722) tensor(107.3000) tensor(59.3000) tensor(44.7344) tensor(119.1000) tensor(60.3000) tensor(49.3703) tensor(165.2000) tensor(119.6000) tensor(27.6029)")
texts.append("tensor(72.1000) tensor(28.6000) tensor(60.3329) tensor(40.3000) tensor(15.7000) tensor(61.0422) tensor(54.4000) tensor(24.6000) tensor(54.7794) tensor(139.4000) tensor(99.7000) tensor(28.4792) tensor(30.8000) tensor(9.2000) tensor(70.1299) tensor(94.3000) tensor(35.7000) tensor(62.1421) tensor(232.6000) tensor(168.7000) tensor(27.4721) tensor(107.9000) tensor(57.8000) tensor(46.4319) tensor(86.2000) tensor(45.5000) tensor(47.2158) tensor(126.6000) tensor(54.2000) tensor(57.1880)")
#320
texts.append("tensor(95.7000) tensor(38.8000) tensor(59.4566) tensor(108.2000) tensor(53.6000) tensor(50.4621) tensor(205.9000) tensor(84.4000) tensor(59.0092) tensor(67.2000) tensor(39.3000) tensor(41.5179) tensor(82.) tensor(34.3000) tensor(58.1707) tensor(119.7000) tensor(51.4000) tensor(57.0593) tensor(78.7000) tensor(27.8000) tensor(64.6760) tensor(96.9000) tensor(42.6000) tensor(56.0372) tensor(119.4000) tensor(69.3000) tensor(41.9598) tensor(87.6000) tensor(27.5000) tensor(68.6073)")
texts.append("tensor(53.) tensor(25.3000) tensor(52.2642) tensor(161.3000) tensor(74.3000) tensor(53.9368) tensor(124.2000) tensor(57.2000) tensor(53.9452) tensor(88.1000) tensor(36.8000) tensor(58.2293) tensor(63.1000) tensor(20.5000) tensor(67.5119) tensor(98.) tensor(60.5000) tensor(38.2653) tensor(100.5000) tensor(34.8000) tensor(65.3731) tensor(80.) tensor(38.4000) tensor(52.) tensor(132.5000) tensor(92.5000) tensor(30.1887) tensor(78.) tensor(33.1000) tensor(57.5641)")
texts.append("tensor(72.6000) tensor(29.5000) tensor(59.3664) tensor(57.) tensor(16.9000) tensor(70.3509) tensor(194.2000) tensor(155.3000) tensor(20.0309) tensor(80.7000) tensor(44.5000) tensor(44.8575) tensor(76.1000) tensor(44.6000) tensor(41.3929) tensor(55.) tensor(30.9000) tensor(43.8182) tensor(98.6000) tensor(57.8000) tensor(41.3793) tensor(163.3000) tensor(74.3000) tensor(54.5009) tensor(111.9000) tensor(55.9000) tensor(50.0447) tensor(91.8000) tensor(23.5000) tensor(74.4009)")
#350
texts.append("tensor(77.5000) tensor(42.1000) tensor(45.6774) tensor(55.2000) tensor(18.2000) tensor(67.0290) tensor(121.6000) tensor(71.2000) tensor(41.4474) tensor(146.) tensor(94.2000) tensor(35.4795) tensor(130.7000) tensor(57.3000) tensor(56.1591) tensor(51.8000) tensor(12.) tensor(76.8340) tensor(55.6000) tensor(21.2000) tensor(61.8705) tensor(91.6000) tensor(43.5000) tensor(52.5109) tensor(55.3000) tensor(23.3000) tensor(57.8662) tensor(93.4000) tensor(36.7000) tensor(60.7066)")
texts.append("tensor(80.1000) tensor(28.8000) tensor(64.0449) tensor(114.3000) tensor(42.6000) tensor(62.7297) tensor(145.3000) tensor(75.8000) tensor(47.8321) tensor(77.1000) tensor(43.5000) tensor(43.5798) tensor(46.4000) tensor(21.1000) tensor(54.5259) tensor(120.2000) tensor(41.6000) tensor(65.3910) tensor(115.2000) tensor(59.1000) tensor(48.6979) tensor(104.) tensor(56.) tensor(46.1538) tensor(98.6000) tensor(34.2000) tensor(65.3144) tensor(62.6000) tensor(36.9000) tensor(41.0543)")
texts.append("tensor(114.1000) tensor(57.1000) tensor(49.9562) tensor(162.8000) tensor(105.8000) tensor(35.0123) tensor(74.9000) tensor(30.6000) tensor(59.1455) tensor(107.1000) tensor(54.8000) tensor(48.8329) tensor(77.4000) tensor(26.) tensor(66.4083) tensor(112.3000) tensor(44.9000) tensor(60.0178) tensor(75.) tensor(33.6000) tensor(55.2000) tensor(95.3000) tensor(37.) tensor(61.1752) tensor(69.4000) tensor(30.9000) tensor(55.4755) tensor(69.3000) tensor(25.2000) tensor(63.6364)")
texts.append("tensor(146.) tensor(71.9000) tensor(50.7534) tensor(78.5000) tensor(30.6000) tensor(61.0191) tensor(100.2000) tensor(34.5000) tensor(65.5689) tensor(79.2000) tensor(28.3000) tensor(64.2677) tensor(132.) tensor(53.2000) tensor(59.6970) tensor(77.) tensor(30.9000) tensor(59.8701) tensor(184.5000) tensor(107.3000) tensor(41.8428) tensor(118.6000) tensor(66.6000) tensor(43.8449) tensor(109.) tensor(61.5000) tensor(43.5780) tensor(86.6000) tensor(45.2000) tensor(47.8060)")
texts.append("tensor(91.9000) tensor(30.6000) tensor(66.7029) tensor(95.8000) tensor(41.7000) tensor(56.4718) tensor(184.8000) tensor(138.8000) tensor(24.8918) tensor(65.7000) tensor(32.9000) tensor(49.9239) tensor(99.4000) tensor(47.5000) tensor(52.2133) tensor(266.6000) tensor(183.8000) tensor(31.0578) tensor(105.) tensor(52.) tensor(50.4762) tensor(95.) tensor(44.2000) tensor(53.4737) tensor(77.) tensor(60.5000) tensor(21.4286) tensor(70.6000) tensor(33.7000) tensor(52.2663)")
texts.append("tensor(96.4000) tensor(38.6000) tensor(59.9585) tensor(121.2000) tensor(56.2000) tensor(53.6304) tensor(72.) tensor(39.8000) tensor(44.7222) tensor(122.5000) tensor(41.7000) tensor(65.9592) tensor(176.4000) tensor(126.4000) tensor(28.3447) tensor(65.7000) tensor(22.7000) tensor(65.4490) tensor(105.7000) tensor(47.1000) tensor(55.4399) tensor(155.7000) tensor(109.2000) tensor(29.8651) tensor(126.9000) tensor(59.5000) tensor(53.1127) tensor(107.4000) tensor(25.1000) tensor(76.6294)")
#410
texts.append("tensor(85.4000) tensor(33.2000) tensor(61.1241) tensor(98.2000) tensor(48.) tensor(51.1202) tensor(137.4000) tensor(57.5000) tensor(58.1514) tensor(57.3000) tensor(33.) tensor(42.4084) tensor(76.1000) tensor(36.4000) tensor(52.1682) tensor(136.6000) tensor(69.) tensor(49.4876) tensor(100.) tensor(55.1000) tensor(44.9000) tensor(130.9000) tensor(69.6000) tensor(46.8296) tensor(87.7000) tensor(55.5000) tensor(36.7161) tensor(110.) tensor(41.) tensor(62.7273)")
texts.append("tensor(117.6000) tensor(57.7000) tensor(50.9354) tensor(79.5000) tensor(32.3000) tensor(59.3711) tensor(76.1000) tensor(33.6000) tensor(55.8476) tensor(119.4000) tensor(85.6000) tensor(28.3082) tensor(124.7000) tensor(51.1000) tensor(59.0217) tensor(95.8000) tensor(27.1000) tensor(71.7119) tensor(101.2000) tensor(32.4000) tensor(67.9842) tensor(76.1000) tensor(34.) tensor(55.3219) tensor(115.5000) tensor(61.9000) tensor(46.4069) tensor(57.1000) tensor(16.5000) tensor(71.1033)")
texts.append("tensor(100.3000) tensor(49.) tensor(51.1466) tensor(104.6000) tensor(45.3000) tensor(56.6922) tensor(86.3000) tensor(38.5000) tensor(55.3882) tensor(87.1000) tensor(36.3000) tensor(58.3238) tensor(83.9000) tensor(28.4000) tensor(66.1502) tensor(70.3000) tensor(43.1000) tensor(38.6913) tensor(95.3000) tensor(47.1000) tensor(50.5771) tensor(90.) tensor(54.8000) tensor(39.1111) tensor(142.5000) tensor(60.6000) tensor(57.4737) tensor(131.2000) tensor(74.6000) tensor(43.1402)")
#440: (and 1st added):
texts.append("tensor(98.6000) tensor(46.8000) tensor(52.5355) tensor(137.3000) tensor(82.7000) tensor(39.7669) tensor(131.3000) tensor(60.8000) tensor(53.6938) tensor(301.5000) tensor(206.2000) tensor(31.6086) tensor(93.) tensor(36.3000) tensor(60.9677) tensor(70.5000) tensor(30.6000) tensor(56.5957) tensor(61.5000) tensor(28.) tensor(54.4715) tensor(99.) tensor(45.3000) tensor(54.2424) tensor(81.8000) tensor(53.3000) tensor(34.8411) tensor(147.6000) tensor(68.7000) tensor(53.4553)")

# convert the text to a tensor
data_camus = np.zeros((450, 3))
l = 0

for i in range(len(texts)):
  tensor = torch.tensor([float(s.split('(')[1].split(')')[0]) for s in texts[i].split()])
  p = 0
  for j in range(10):
    data_camus[l][0] = tensor[p].item()
    data_camus[l][1] = tensor[p+1].item()
    data_camus[l][2] = tensor[p+2].item()
    p += 3
    l += 1

vedscamus = []
vesscamus = []
for c in range(450):
  vedscamus.append(data_camus[c][0])
  vesscamus.append(data_camus[c][1])

veds = []
vess = []
i=0
greenp0=[]
greenp1=[]

##put in veds2 the vector with all the values (ved, ves)_i obtained with create_dt8.py
veds2 = 

n0=6
n1=4
n2=4
n3=8
n4=8
n5=8
n6=1
tcs = np.linspace(0.5, 2., n0)
startvs = np.linspace(15., 400., n1)
startpaos = np.linspace(5., 150., n2)
rcs = np.linspace(0.05, 4., n3)
emaxs = np.linspace(0.5, 50., n4)
emins = np.linspace(0.02, 0.3, n5)
vds = np.linspace(4., 40., n6)

i = 0
for Tc in tcs:
  for start_v in startvs:
    for start_pao in startpaos:
      for Rc in rcs:
        for Emax in emaxs:
          for Emin in emins:
            for Vd in vds:
              ved =  veds2[i][0]
              ves = veds2[i][1]
              ped = ved * Emin
              pes = ves * Emax

              if (ped>4. and pes<150.):
                veds.append(ved)
                vess.append(ves)
              else:
                greenp0.append(ved)
                greenp1.append(ves)

              i+=1

plt.scatter(greenp0, greenp1, color='g')
plt.scatter(vedscamus, vesscamus, color='b')
plt.scatter(veds, vess, color='r')
plt.show()
