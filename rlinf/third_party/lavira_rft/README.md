# LaViRA-RFT vendored core

`source/` is a byte-for-byte copy of the navigation core from:

- repository: `/home/lhx/workspace/lavira-code`
- branch: `feat/merged-la-va`
- commit: `b3d6c35067ad731a16ec096b698ddfc2f5c91af6`

Do not edit files under `source/`. RLinf compatibility belongs in sibling
adapter modules. The runtime loader supplies Python 3.11 compatibility aliases
for dependencies that are not used by mapping or FMM; it does not rewrite the
algorithm bodies.

Vendored ownership is deliberately limited to mapping, FMM/Policy, map/depth
utilities, RepViT-SAM construction, and their import dependencies. Habitat
environment ownership and batched model inference remain in RLinf.
