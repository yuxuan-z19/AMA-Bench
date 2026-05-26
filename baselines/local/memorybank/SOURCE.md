# MemoryBank Source Snapshot

- Source repo: https://github.com/zhongwanjun/MemoryBank-SiliconFriend
- Source commit: `cf61c4196e4cfdb0f2b7a0316249fa40312dc3a9`
- Source paper: https://arxiv.org/abs/2305.10250
- Local role: source reference plus official eval data snapshot.

The official implementation targets SiliconFriend chatbot sessions.  The
AMA-Bench adapter in `src/method/memorybank_method.py` preserves the
MemoryBank mechanisms needed for benchmark inference: raw memory storage,
event summarization, dense retrieval, retention metadata, and recall
reinforcement.
