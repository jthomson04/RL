# Dynamo generation design

The Dynamo backend supports two strict ownership modes selected by the shape of
`policy.generation.dynamo_cfg`:

- Managed Slurm mode owns a fixed vLLM fleet inside the Ray allocation.
- Kubernetes mode connects to an externally owned
  `DynamoGraphDeployment` (DGD).

Both modes share generation, token wrapping, metrics, native vLLM NCCL refit,
and cache invalidation. Only service discovery, placement, and lifecycle
ownership differ.

## Ownership and placement

Constructing `ManagedDynamoRuntime` is inert. Its explicit `start()` method
allocates ports, launches etcd and NATS JetStream, creates one Ray-managed
`dynamo.vllm` process per model-parallel group, and starts the frontend. A
worker group must fit on one node. Its world size is derived from vLLM tensor
parallelism times pipeline parallelism; expert parallelism must be either one
or equal to tensor parallelism.

Startup completes only after the frontend sees the same fixed membership at
the generation and RL endpoints and advertises the configured model. Worker
handles are recorded before readiness checks so partial startup failures can
be torn down. Shutdown is idempotent and guards the frontend, worker pool,
NATS, etcd, and temporary state independently.

## Generation state

Both GRPO trainers use `DynamoGeneration.generate_async()` against the managed
frontend. NeMo-Gym traffic passes through a process-local token wrapper. It
uses the policy tokenizer, preserves caller `nvext.extra_fields`, adds Dynamo
engine metadata, and translates rendered multi-turn prefixes back to the exact
caller token IDs.

Serialized rollout copies contain only frontend URLs and immutable worker
admin endpoints. They cannot own or stop services. Those endpoints are enough
for AREAL-style post-refit cache invalidation; Magistral keeps its existing
driver-side invalidation lifecycle.

## Weight refit

Dynamo uses `CollectiveWeightSynchronizer`. If each engine has world size `E`,
worker `i` starts at rank `training_world_size + i * E`. The policy sender uses
vLLM's peer initialization and its fixed packed-transfer geometry: two 1-GiB
buffers. The isolated vLLM environment validates the same constants before a
worker starts.

Generation is drained before refit. The worker then runs vLLM's native
`start_weight_update`, `update_weights`, and `finish_weight_update` transaction.
KV-cache invalidation stays outside the generic synchronizer because GRPO's
cache mode determines where it runs.

## Dependency isolation

`BUILD_DYNAMO=1` adds a Python 3.12 `/opt/dynamo_venv` to the standard image.
It contains only `ai-dynamo[vllm]==1.3.0.post1`, its pinned vLLM 0.23.0, etcd,
and NATS. NeMo-RL's normal Ray and engine environments are unchanged; the
standard NeMo-RL vLLM environment currently uses vLLM 0.25.1.

vLLM 0.23.0 predates PR #44814, which fixes layerwise reload accounting for
composed loaders. The installer asserts the exact vLLM version, checks and
applies the backport, and records upstream merge commit
`c9e5bf813530fb9ce06024e075da0f520b0718c8` in
`/opt/dynamo_venv/VLLM_BACKPORTS`. Remove the backport only after Dynamo pins a
vLLM release containing that fix. At that point delete the patch, application
logic, marker assertion, and backport text rather than rebasing the patch.

## Kubernetes DGD runtime

The external runtime does not create, scale, or delete serving workers.
`nrl-k8s` creates or reuses the DGD, waits for the Dynamo operator to report
it ready, and only then starts the Ray training workload. The Dynamo operator
and `DynamoGraphDeployment` CRD must already be installed in the target
cluster.

The strict DGD config shape is:

```yaml
policy:
  generation:
    backend: dynamo
    dynamo_cfg:
      engine_world_size: 1
      dgd_name: my-dgd
      frontend_url: null
      namespace: null
      frontend_port: 8000
      dyn_system_port: 9090
      request_timeout_s: 900.0
      discovery_timeout_s: 15.0
      control_timeout_s: 600.0
      exclude_tools_when_tool_choice_none: true
      metrics_include_prefixes: null
      metrics_exclude_prefixes: null
```

The checked-in recipes omit `dgd_name` because their nrl-k8s entrypoints
inject the concrete deployment name. `frontend_url` is an escape hatch for a
reachable nonstandard frontend. NCCL refit still requires `dgd_name`, because
worker discovery uses that deployment's frontend health endpoint. When
`namespace` is null, NeMo-RL reads the pod's projected service-account
namespace; it never silently guesses `default`.

The DGD manifest owns the inference engine arguments. Keep
`engine_world_size` and the recipe's vLLM geometry consistent with the DGD.
Each vLLM worker must enable the RL routes and native NCCL transfer backend:

```yaml
args:
  - --enable-rl
  - --weight-transfer-config
  - '{"backend":"nccl"}'
```

### Fixed worker discovery

The runtime derives
`http://<dgd>-frontend.<namespace>.svc.cluster.local:<frontend_port>/v1`
for rollout requests. It reads `GET /health` from the same frontend, filters
registrations to the deployment namespace plus `component: backend` and
`endpoint: rl`, deduplicates by `instance_id`, and converts the advertised
pod address to the worker admin port.

The ordered `(instance_id, system_url)` fleet is frozen at setup. Before
refit operations the runtime rediscovers membership and fails if any worker
scaled, restarted, disappeared, or changed address. Restart training after a
DGD membership change so a new NCCL collective can be established.

If there are `N` workers and each has `engine_world_size = E`:

```text
inference_world_size = N * E
world_size = training_world_size + inference_world_size
worker[i].rank_offset = training_world_size + i * E
```

The shared `DynamoRefitChannel` performs the same native vLLM transaction
used by managed mode. The external runtime's shutdown is intentionally a
no-op; Kubernetes remains the DGD lifecycle owner.

### nrl-k8s RayJob ordering and cleanup

For ephemeral runs, nrl-k8s creates the RayJob with `spec.suspend: true` and
waits for `jobDeploymentStatus: Suspended`. It creates the DGD and DRA
prerequisites with the RayJob as their garbage-collection owner, waits for the
DGD's current generation to report `Ready=True`, and then resumes the
RayJob.

After KubeRay reports the generated RayCluster, nrl-k8s reparents only
resources created for this run to that RayCluster. Reused DGDs and DRA
resources are never adopted. If reparenting fails, the RayJob owner remains as
a TTL-based cleanup fallback.

The four GB300 examples are under `infra/nrl_k8s/examples/dynamo/`. Validate
and render any recipe/infra pair before launching:

```bash
RECIPE=infra/nrl_k8s/examples/dynamo/V1/grpo_math_1b_dynamo_nccl.yaml
INFRA=infra/nrl_k8s/examples/dynamo/V1/grpo_math_1b_dynamo_nccl.gb300.infra.yaml

nrl-k8s check "$RECIPE" --infra "$INFRA"
nrl-k8s run "$RECIPE" --infra "$INFRA" --rayjob --dry-run
nrl-k8s run "$RECIPE" --infra "$INFRA" --rayjob --no-wait
```

See [Managed Dynamo generation on Slurm](../guides/dynamo-generation.md) for
build, configuration, and launch instructions.
