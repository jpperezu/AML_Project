OPS = {
  'unconditional' : lambda x, guidance: x.chunk(2)[0],
  'conditional' : lambda x, guidance: x.chunk(2)[1],
  'guided' : lambda x, guidance: x.chunk(2)[0] + guidance * (x.chunk(2)[1] - x.chunk(2)[0])
}