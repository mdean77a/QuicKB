# Apple Silicon Compatibility

I was able to run this easily on an M3 Macbook Pro by making the following changes. I don't know if there is a way to alter the yaml file dynamically and when I tried to add any parameters I got a Pydantic error, because extras were forbidden.

## train.py

Change device setup to read:

`
  device = "mps" if torch.mps.is_available() else "cpu"
`

## config.yaml

Changed training arguments:

`
  batch_size: 16         # (Not sure if this was necessary, was suggested by Grok)
  optim:"adamw_torch"    # fused not supported on mps
  tf32: false
  bf16: false
`
