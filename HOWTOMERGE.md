HOWTOMERGE (clear remotes + push safety)
0) Clone the gfx906 fork you start from
git clone https://github.com/iacopPBK/llama.cpp-gfx906.git
cd llama.cpp-gfx906
1) Rename origin to make it obvious what it is

After clone, origin points at iacopPBK. Rename it:

git remote rename origin iacopPBK
2) Add the two other remotes
git remote add upstream https://github.com/ggml-org/llama.cpp.git
git remote add kinchahoy git@github.com:kinchahoy/llama.cpp.git
3) Disable pushing everywhere except kinchahoy
git remote set-url --push iacopPBK DISABLE
git remote set-url --push upstream DISABLE
# kinchahoy remains push-enabled (do nothing)

(Optional) Make plain git push go to kinchahoy by default:

git config --local remote.pushDefault kinchahoy
git config --local push.default current

Sanity check:

git remote -v
git config --local --get remote.pushDefault
4) Fetch everything
git fetch --all --prune
5) Create or update your gfx906 rebase branch on top of upstream head

First-time create:

git checkout -b gfx906-rebased upstream/master

Or if it already exists:

git checkout gfx906-rebased
git rebase upstream/master
6) Resolve known conflict (mmq)

File: ggml/src/ggml-cuda/mmq.cuh
Conflict: MMQ_ITER_K / MMQ_NWARPS macro block

Decision:

keep upstream MMQ_ITER_K_MXFP4_FP4 512

for HIP builds use GFX906_MMQ_ITER_K and GFX906_MMQ_NWARPS

for non-HIP use upstream defaults MMQ_ITER_K 256, MMQ_NWARPS 8

Then:

git add ggml/src/ggml-cuda/mmq.cuh
git rebase --continue
7) Build/test

(run your usual build steps)

8) Push to your fork (only pushable remote)
git push -u kinchahoy gfx906-rebased

If this branch already exists on GitHub and you rebased:

git push --force-with-lease kinchahoy gfx906-rebased
Resulting remote layout (what you want)

upstream = ggml-org (fetch only)

iacopPBK = starter fork (fetch only)

kinchahoy = your fork (fetch + push)
