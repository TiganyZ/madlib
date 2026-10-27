# Fine-tuning GRACE over several SEI systems

The `grace_lifco` pipeline, run over several small systems that together cover
a graphite | SEI | electrolyte interface. After that there is one labelling
pass and one fine-tune on the merged selection. `CATION=Na` (default) or
`CATION=Li` picks the alkali metal.

| system | contents |
|---|---|
| `inorganic` | MF, M₂CO₃ and M₂O, plus MF and M₂O crystals |
| `organic` | M ethylene dicarbonate, M methyl carbonate, MF |
| `electrolyte` | EC, DMC and MPF₆ |
| `graphite_electrolyte` | an H-terminated graphite edge plus electrolyte, and graphite crystals |
| `graphite_sei` | the same edge plus SEI components |

```sh
./run_all.sh                 # stages 01-06 per system, then merge, label, fine-tune, validate
CATION=Li ./run_all.sh
```

- **Per system.** Each system runs `grace_lifco` stages 01–06 with its own
  `params/<system>.sh`, which sets the species, the melt temperature and the
  MAD temperatures.
- **Merge.** `stages/06_merge.sh` joins the selections and prefixes each frame's
  name with its system.
- **Once, on the merged set.** `grace_lifco` stages 07–09 label, fine-tune and
  validate, in `runs/sei_<cation>/combined/`.

`data/molecules.xyz` holds the molecular units. `stages/structures.py` packs
them at random and pushes them apart until no two units are closer than 2 Å.

The labels are PBE without dispersion. Add dispersion at run time with
`pair_style hybrid/overlay grace/fs dispersion/d3`.
