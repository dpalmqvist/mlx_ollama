# Distributed Inference with olmlx

olmlx supports distributed inference across multiple Apple Silicon devices to scale inference capacity. This feature allows you to distribute inference workload across multiple devices or nodes.

## Prerequisites

- SSH access to worker nodes
- SSH keys configured for passwordless access
- All nodes must have the same model installed
- Network connectivity between nodes

## Configuration

### Hostfile Setup

Create a hostfile that defines your distributed cluster:

```json
{
  "hosts": ["localhost", "worker1", "worker2"],
  "model": "mlx-community/Llama-3.2-3B-Instruct-4bit"
}
```

### Environment Variables

Set the following environment variables to enable distributed inference:

```bash
export OLMLX_EXPERIMENTAL_DISTRIBUTED=true
export OLMLX_EXPERIMENTAL_DISTRIBUTED_HOSTFILE=~/.olmlx/hostfile.json
export OLMLX_EXPERIMENTAL_DISTRIBUTED_BACKEND=ring
export OLMLX_EXPERIMENTAL_DISTRIBUTED_PORT=32323
export OLMLX_EXPERIMENTAL_DISTRIBUTED_SIDEBAND_PORT=32400
export OLMLX_EXPERIMENTAL_DISTRIBUTED_SECRET="secret"
```

### SSH Configuration

Configure SSH access to worker nodes:

```bash
ssh-keyscan worker1 >> ~/.ssh/known_hosts
ssh-keyscan worker2 >> ~/.ssh/known_hosts
```

## Usage

1. **Create the hostfile**:
   ```bash
   mkdir -p ~/.olmlx
   echo '{"hosts": ["localhost", "worker1", "worker2"], "model": "mlx-community/Llama-3.2-3B-Instruct-4bit"}' > ~/.olmlx/hostfile.json
   ```

2. **Set environment variables**:
   ```bash
   export OLMLX_EXPERIMENTAL_DISTRIBUTED=true
   export OLMLX_EXPERIMENTAL_DISTRIBUTED_HOSTFILE=~/.olmlx/hostfile.json
   export OLMLX_EXPERIMENTAL_DISTRIBUTED_SECRET="secret"
   ```

3. **Start the server**:
   ```bash
   olmlx serve
   ```

## Security Considerations

- The distributed mode uses plaintext TCP communication between nodes
- Use a shared secret for authentication between nodes
- For production use, consider using SSH tunnels or VPNs to secure communication

## Limitations

- Distributed inference requires all nodes to have the same model installed
- The coordinator node must be able to SSH to all worker nodes
- Workers must have sufficient GPU memory to handle their share of the workload
- Distributed inference is experimental and may have stability issues

## Troubleshooting

If you encounter issues with distributed inference:

1. **Check SSH connectivity**:
   ```bash
   ssh worker1
   ssh worker2
   ```

2. **Verify SSH keys are configured correctly**:
   ```bash
   ssh worker1 'echo "SSH works"'
   ```

3. **Check that all nodes have the same model installed**:
   ```bash
   olmlx model list
   ```

4. **Check logs for errors**:
   ```bash
   cat ~/.olmlx/olmlx.log
   ```