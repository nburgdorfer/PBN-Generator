# pbn-generator

Paint-by-numbers image generator available as both a Python package and a CLI.

## Python API

```python
from pbn_generator import InputConfig, PBNConfig, PBNGenerator

config = PBNConfig(
    input=InputConfig(
        image="data/hills.jpg",
        palette="palettes/palette_1.yaml",
    )
)
result = PBNGenerator(config).run()

print(result.outline_path)
```

## CLI

```bash
pbn-generator input.image=data/hills.jpg input.palette=palettes/palette_1.yaml
```

Hydra is installed as a required dependency so CLI overrides work immediately after
`pip install pbn-generator`.
