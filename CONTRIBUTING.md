# Contributing

Contributions are welcome when they improve correctness, reproducibility,
documentation, or usability.

Before opening a pull request:

```bash
python3 scripts/preflight_github.py
python3 scripts/run_tests.py
```

For physics or numerical changes:

1. State the physical/numerical assumption being changed.
2. Add or update a regression test whenever practical.
3. Include a quantitative comparison against an analytic result, previous
   result, RADIA result, or published/experimental benchmark when relevant.
4. Do not describe a prototype geometry as facility-certified without a
   documented facility-specific benchmark.
5. Keep generated data and large result files out of normal commits.

Pull requests should be focused and should explain both the implementation
change and any expected change in numerical output.
