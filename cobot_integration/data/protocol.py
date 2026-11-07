# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from RoboLab tools/pi05/protocol.py; see LICENSE.protocol.
"""Bounded, pickle-free messages over a job-local Unix socket."""
import io
import json
import struct
import numpy as np

MAX_BYTES = 16 * 1024 * 1024


def receive(sock):
    def exact(n):
        chunks = bytearray()
        while len(chunks) < n:
            part = sock.recv(n - len(chunks))
            if not part:
                raise EOFError('Policy connection closed')
            chunks.extend(part)
        return bytes(chunks)
    size, = struct.unpack('!I', exact(4))
    if not 0 < size <= MAX_BYTES:
        raise ValueError('Invalid policy message size')
    with np.load(io.BytesIO(exact(size)), allow_pickle=False) as packet:
        metadata = json.loads(bytes(packet['metadata']).decode())
        arrays = {k: packet[k].copy() for k in packet.files if k != 'metadata'}
    return metadata, arrays


def send(sock, metadata, arrays=None):
    if any(np.asarray(v).dtype.hasobject for v in (arrays or {}).values()):
        raise ValueError('Object arrays are not allowed in policy messages')
    stream = io.BytesIO()
    np.savez(stream, metadata=np.frombuffer(json.dumps(metadata, allow_nan=False).encode(), dtype=np.uint8),
             **(arrays or {}))
    data = stream.getvalue()
    if len(data) > MAX_BYTES:
        raise ValueError('Policy message too large')
    sock.sendall(struct.pack('!I', len(data)) + data)
