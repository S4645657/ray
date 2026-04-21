# Session-Aware Consistent-Hash Router for Ray Serve

## Goal

Route requests from the same `session_id` to the same replica, with minimal
reshuffling when the replica set changes. Falls back to standard Power-of-Two
routing when no `session_id` is present on the request.

## Prior art

### vLLM router — [`consistent_hash.rs`](https://github.com/vllm-project/router/blob/dbdf3201d4fb88e84d7a185dd85e5781b92ee39f/src/policies/consistent_hash.rs)

- Per-policy hash ring keyed by worker URL.
- `VIRTUAL_NODES_PER_WORKER = 160`.
- `BTreeMap<u64, String>`; lookup is
  `ring.range(h..).next().or_else(|| ring.iter().next())` (wraps around).
- Change-detection via cached `current_workers: Vec<String>`; skip rebuild
  if the set hasn't changed.
- Virtual-node key: `f"{worker_url}:{i}"`.
- Hash: Facebook mcrouter's `fbi_hash` (`furc_hash` → 23-bit →
  `MurmurHash64A` expansion to 64-bit).
- Hash-key extraction priority: headers → body → fallback.
- On unhealthy: fall back to `healthy_indices[0]`.

### Cassandra — `Murmur3Partitioner` + `TokenMetadata`

- **MurmurHash3_x64_128**, take low 64 bits as the signed token
  (`LongToken`, range `[Long.MIN_VALUE, Long.MAX_VALUE]`). Better avalanche
  than `MurmurHash64A` at nearly identical cost.
- **Per-node `num_tokens`** — each node owns a configurable count of
  tokens (historically 256, now ~16). Enables weighted ownership for
  heterogeneous nodes.
- **Precomputed replication endpoints.** Cassandra doesn't walk the ring
  at read time; `AbstractReplicationStrategy` precomputes each token's
  ordered list of replica successors at topology-change time. Request-
  time lookup is O(log n) primary lookup + O(1) fallback read.
- `ConcurrentSkipListMap<Token, InetAddress>` for the sorted structure.

---

## Packaging: a standalone router, not a mixin

Follow the `CapacityQueueRouter` pattern
(`python/ray/serve/experimental/capacity_queue_router.py`): ship a new
router class under `python/ray/serve/experimental/consistent_hash_router.py`,
with the same inheritance shape as `CapacityQueueRouter`:

```python
class ConsistentHashRouter(LocalityMixin, MultiplexMixin, RequestRouter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ring_hashes: List[int] = []
        self._ring_replicas: List[ReplicaID] = []
        self._replica_set_snapshot: FrozenSet[ReplicaID] = frozenset()
        self._uid_to_replica: Dict[str, RunningReplica] = {}

    def initialize_state(self, **kwargs):
        self._virtual_nodes = kwargs.get("virtual_nodes", 160)
        self._fallback_replicas = kwargs.get("fallback_replicas", 2)
        self._session_header = kwargs.get("session_header", "X-Session-Id")

    def update_replicas(self, replicas: List[RunningReplica]):
        super().update_replicas(replicas)
        self._rebuild_ring_if_changed(replicas)

    async def choose_replicas(
        self,
        candidate_replicas: List[RunningReplica],
        pending_request: Optional[PendingRequest] = None,
    ) -> List[List[RunningReplica]]:
        session_id = self._extract_session_id(pending_request)
        if session_id is None:
            # Fallback to P2C when the caller didn't pass a session.
            return await PowerOfTwoChoicesRequestRouter.choose_replicas(
                self, candidate_replicas, pending_request
            )
        return [self._lookup_ring(session_id, candidate_replicas)]
```

Users opt in via `RequestRouterConfig`:

```python
request_router_class=(
    "ray.serve.experimental.consistent_hash_router:ConsistentHashRouter"
),
request_router_kwargs={"virtual_nodes": 160, "fallback_replicas": 2},
```

## Where the ring lives

**Each `ConsistentHashRouter` instance owns its own hash ring.**

Ray Serve already instantiates one `RequestRouter` per `Router`, and there
is one `Router` per `DeploymentHandle`. There is *no* centralized ring
service.

Consequences:

- Two routers for the same deployment compute their rings independently.
  They converge because they both receive the same `update_replicas()`
  calls from the controller's long-polled replica set — inputs are
  globally consistent, so outputs are too, as long as each router uses
  the same hashing algorithm and the same virtual-node count.
- The ring holds no per-session state. A session that has never been seen
  still maps deterministically to a replica the first time it is hashed.
  Router restarts do not break affinity: a new router rebuilds the same
  ring from the same replica set and the same session hashes to the same
  owner.
- Rings are not shared across deployments. A deployment upgrade that
  instantiates a new router rebuilds the ring from scratch — which is
  fine, because the replica set itself changed.

The ring is rebuilt inside the `update_replicas()` lifecycle hook
(`request_router.py:871-924`) — the same hook `MultiplexMixin` uses to
maintain its `model_id → replicas` map.

---

## Ring representation

- Each replica contributes `V` **virtual nodes** to the ring (default
  `V=160`, matching vLLM's `VIRTUAL_NODES_PER_WORKER`; tunable via
  `request_router_kwargs={"virtual_nodes": 160}`).
- A virtual node's ring position is
  `hash(f"{replica_id.unique_id}:{vnode_index}")`, where
  `vnode_index ∈ [0, V)` — same key format vLLM uses.
- The ring is stored as two parallel arrays kept sorted by hash:
  `ring_hashes: List[int]` and `ring_replicas: List[ReplicaID]`.
  Lookup is `idx = bisect.bisect_right(ring_hashes, session_hash)`; if
  `idx == len(ring_hashes)`, wrap to `0`. (Equivalent to vLLM's
  `BTreeMap::range(h..).next().or_else(first)`.)
- `ReplicaID.unique_id` (not actor id) is the stable hash key: it survives
  actor restarts and does not change under scale events
  (`common.py:59-136`).
- **Change detection.** Cache the previous `Set[ReplicaID]` and skip the
  rebuild if it's unchanged (vLLM's `current_workers` optimization).
  `update_replicas()` is called on every long-poll tick even when nothing
  changed — important to avoid thrash.

The ring is immutable once built — `update_replicas()` constructs a new ring
and swaps it in with a single assignment, so concurrent `choose_replicas()`
callers always see a consistent ring.

---

## Hashing algorithm

Both the virtual-node positions and the session lookups use the same hash.

- **Algorithm:** `MurmurHash64A` with a fixed seed (`0xF9B4CA77`). 64-bit
  output, deterministic across processes, well-distributed for short
  string keys. Available in Python via `mmh3.hash64()`.
- **Fallback (no mmh3 installed):** `hashlib.blake2b(digest_size=8)` —
  stdlib, deterministic, slower but still fine for routing-rate volumes.
- **Why not `fbi_hash` like vLLM?** vLLM's `fbi_hash` composes `furc_hash`
  (a 23-bit jump-consistent-hash variant) with `MurmurHash64A` expansion
  to 64 bits. That composition exists for mcrouter compatibility, not
  because furc_hash has better distribution. We have no mcrouter
  compatibility requirement, so we use `MurmurHash64A` directly — simpler,
  same statistical properties on a 64-bit ring.
- **Do not use `hash()` / `PYTHONHASHSEED`.** Python's builtin `hash` is
  randomized per process and would give different rings in different
  routers — breaking cross-router affinity for the same session.
- **Session-id hashing:** `murmur64a(session_id.encode("utf-8"))`. No
  salting — the ring and the session must inhabit the same hash space.

**Load distribution:** with `V=160` virtual nodes per replica and `R`
replicas, the standard deviation of load across replicas is
`~1/sqrt(V·R)` ≈ 2.5% at R=10. Increase `V` for smoother distribution at
the cost of ring-build time (O(R·V log(R·V))) and memory (~16 bytes per
vnode — at R=50, V=160, that's 128 KB).

---

## Replica joins

When `update_replicas()` is called and replica `r_new` is in the new set
but not the old:

1. A new ring is built from scratch from the new replica set. (Full rebuild
   is O(R·V log(R·V)); for R=50, V=100, this is ~5k entries — microseconds.
   Incremental updates are not worth the complexity.)
2. `r_new` contributes `V` virtual nodes at hash positions determined by
   `xxh64(r_new.unique_id + "#i")` for `i ∈ [0, V)`.
3. Each of `r_new`'s virtual nodes **steals a hash range** from its
   clockwise-next neighbor. Sessions whose hash falls in a stolen range are
   now owned by `r_new`.
4. Expected fraction of sessions remapped: **`1/R_new`**, where `R_new` is
   the replica count after the join. Sessions not in a stolen range stay
   with their previous owner.
5. In-flight requests are unaffected — they were already dispatched to a
   specific replica before the ring change. Only *future* requests consult
   the new ring.

There is no "warm-up" or handoff. If the application needs to warm per-session
state on `r_new`, that is an application-layer concern (lazy load on first
request).

---

## Replica leaves

When replica `r_old` is in the old set but not the new (scale-down,
crash, rolling upgrade):

1. A new ring is built from the new replica set; `r_old`'s `V` virtual
   nodes are absent.
2. Each hash range previously owned by `r_old`'s vnodes is absorbed by the
   clockwise-next vnode on the new ring. Those ranges redistribute across
   the *remaining* replicas proportionally to their vnode density.
3. Expected fraction of sessions remapped: **`1/R_old`**, where `R_old` is
   the replica count before the leave. Sessions owned by other replicas are
   untouched.
4. **In-flight requests** already sent to `r_old` follow the normal
   draining / retry behavior in `request_router.py` — if `r_old` is
   unreachable, the outer retry loop (`_choose_replicas_with_backoff`,
   lines 1145-1214) calls `choose_replicas()` again and the new ring
   returns a different owner.

### Graceful vs. abrupt departures

- **Graceful (rolling upgrade, scale-down):** the controller marks
  `r_old` as draining, then removes it from the replica set. The router
  sees the removal via `update_replicas()` and stops routing new sessions
  to it. Existing requests on `r_old` finish on `r_old`.
- **Abrupt (crash):** `r_old` vanishes from the replica set on the next
  long-poll. Until then, the router may still hash a session to `r_old`
  and the request fails; the retry loop then re-hashes against the new ring.
  This is the same failure mode the existing P2C router has — we do not
  make it worse.

### Unhealthy replica on the current ring

If the ring returns `r` but `r` is currently in a bad state (circuit
breaker tripped, queue full, actor unreachable):

- vLLM's router falls back to `healthy_indices[0]` — simple, but it
  concentrates every unhealthy-session hit on one replica and throws away
  affinity completely.
- We instead **walk the ring clockwise** to the next distinct replica,
  returning up to `K` candidates in ranked order. This preserves
  neighborhood locality (likely same node / same rack under
  `LocalityMixin`) and distributes fallback load proportionally to vnode
  density rather than piling it on index 0.

---

## Handling the hot-session problem

Strict consistent hashing can overload a single replica if one `session_id`
is very hot. Two mitigations, both configurable:

1. **Fallback candidates.** `choose_replicas()` returns a ranked list. We
   return the ring owner as the top candidate and the next `K` clockwise
   vnodes' replicas (deduplicated) as fallbacks. The retry loop already
   walks this list when the top choice is unavailable or at queue capacity.
   Default `K=2`.
2. **Bounded-load consistent hashing (optional, later).** Cap each
   replica's in-flight load at `c · avg_load` (c≈1.25); if the ring owner
   is at cap, walk to the next vnode. This preserves affinity on the common
   path but prevents pathological overload. Not in v1.

---

## Missing session_id

If the request has no session_id, `choose_replicas()` delegates to
`PowerOfTwoChoicesRequestRouter.choose_replicas(self, ...)` — same pattern
`CapacityQueueRouter` uses for its Pow2 fallback path. This keeps
`ConsistentHashRouter` usable as a drop-in replacement for deployments
that mix session-aware and session-less traffic.

---

## Summary table

| Event              | Ring action              | Sessions remapped       | Notes                               |
| ------------------ | ------------------------ | ----------------------- | ----------------------------------- |
| Replica joins      | Rebuild, new V vnodes    | ~`1/R_new`              | New replica steals ranges           |
| Replica leaves     | Rebuild, V vnodes gone   | ~`1/R_old`              | Ranges redistributed to neighbors   |
| Router restarts    | Rebuild from replica set | 0 (deterministic hash)  | Requires stable hash algorithm      |
| Deployment upgrade | New router, new ring     | All (new replica set)   | Expected — replicas changed         |
| Missing session_id | Bypass ring              | N/A                     | Falls back to P2C                   |

---

## Open questions for review

1. **Where does `session_id` come from on the wire?** Header (`X-Session-Id`)
   for HTTP and gRPC metadata for gRPC, extracted in the proxy the same way
   `multiplexed_model_id` is extracted? Or should we also accept it as a
   kwarg on `handle.remote(..., session_id=...)`?
2. **Should V be per-replica-weighted** to support heterogeneous replicas
   (e.g. different GPU sizes)? Not in v1, but the API should leave room.
3. **Do we need bounded-load in v1?** Or is the K-fallback sufficient for
   the initial release?
