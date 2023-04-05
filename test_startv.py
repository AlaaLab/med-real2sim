err = 0
ntot = 500
for i in range(ntot):
  start_v = random.uniform(40., 400.)
  start_pao = random.uniform(50., 100.)
  Emax = random.uniform(1., 4.)
  Emin = random.uniform(0.02, 0.10)
  Rc = random.uniform(0.025, 0.075) #at least check in the interval closer to the interpolated one ([0.025, 0.074])
  Rs = random.uniform(0.2, 1.8)
  Cs = random.uniform(0.6, 2.0)

  ved1 = interp([start_v, start_pao, Emax, Emin, Rc, Rs, Cs])
  v2 = interp([start_v + 0.001, start_pao, Emax, Emin, Rc, Rs, Cs])

  vedreal = f(start_v, start_pao, Emax, Emin, Rc, Rs, Cs)
  vreal2 = f(start_v + 0.001, start_pao, Emax, Emin, Rc, Rs , Cs)

  gradsim = (v2 - ved1) / 0.001
  gradreal = (vreal2 - vedreal) / 0.001

  err += abs(gradsim - gradreal) / abs(gradreal)

print("Average error in dV_ED/dstartv: ", err/ntot)
