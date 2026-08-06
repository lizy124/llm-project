#!/usr/bin/env python3
import importlib
import json
from pathlib import Path

ROOT = Path('/vllm-workspace/vllm-ascend')
BASE = ROOT / 'vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store'

EXPECTED_FILES = [
    'backend/__init__.py',
    'backend/backend.py',
    'store_utils/__init__.py',
    'store_utils/ascend_store_connector.py',
    'store_utils/config_data.py',
    'store_utils/coordinator.py',
    'store_utils/kv_transfer.py',
    'store_utils/pool_scheduler.py',
    'store_utils/pool_worker.py',
    'mooncake_backend/__init__.py',
    'mooncake_backend/mooncake_backend.py',
    'memcache_backend/__init__.py',
    'memcache_backend/memcache_backend.py',
    'yuanrong_backend/__init__.py',
    'yuanrong_backend/yuanrong_backend.py',
    'ucm_connector/__init__.py',
    'ucm_connector/ucm_connector.py',
    'lmcache_ascend_connector/__init__.py',
    'lmcache_ascend_connector/lmcache_ascend_connector.py',
]

REMOVED_OLD_FILES = [
    'ascend_store_connector.py',
    'config_data.py',
    'coordinator.py',
    'kv_transfer.py',
    'pool_scheduler.py',
    'pool_worker.py',
    'memcache_backend.py',
    'mooncake_backend.py',
    'yuanrong_backend.py',
    'ucm_connector.py',
    'lmcache_ascend_connector.py',
]

REQUIRED_IMPORTS = [
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.store_utils.ascend_store_connector',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.store_utils.config_data',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.store_utils.coordinator',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.store_utils.kv_transfer',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.store_utils.pool_scheduler',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.store_utils.pool_worker',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mooncake_backend.mooncake_backend',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.memcache_backend.memcache_backend',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.yuanrong_backend.yuanrong_backend',
]
OPTIONAL_IMPORTS = [
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ucm_connector.ucm_connector',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.lmcache_ascend_connector.lmcache_ascend_connector',
]


def try_import(name):
    try:
        module = importlib.import_module(name)
        return {'module': name, 'ok': True, 'file': getattr(module, '__file__', None)}
    except Exception as exc:
        return {'module': name, 'ok': False, 'error': repr(exc)}


def registry_target(loader):
    if loader is None or not getattr(loader, '__closure__', None):
        return None
    cells = [cell.cell_contents for cell in loader.__closure__]
    return {'class': cells[0], 'module': cells[1]}


def main():
    from vllm_ascend.distributed.kv_transfer import register_connector
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import backend_map

    register_connector()
    registry = {
        name: registry_target(KVConnectorFactory._registry.get(name))
        for name in ['AscendStoreConnector', 'MooncakeConnectorStoreV1', 'UCMConnector', 'LMCacheAscendConnector']
    }
    result = {
        'base': str(BASE),
        'expected_files': {path: (BASE / path).exists() for path in EXPECTED_FILES},
        'old_root_files_absent': {path: not (BASE / path).exists() for path in REMOVED_OLD_FILES},
        'required_imports': [try_import(name) for name in REQUIRED_IMPORTS],
        'optional_imports': [try_import(name) for name in OPTIONAL_IMPORTS],
        'backend_map': backend_map,
        'connector_registry': registry,
    }
    result['ok'] = (
        all(result['expected_files'].values())
        and all(result['old_root_files_absent'].values())
        and all(item['ok'] for item in result['required_imports'])
        and backend_map['mooncake']['path'].endswith('mooncake_backend.mooncake_backend')
        and backend_map['memcache']['path'].endswith('memcache_backend.memcache_backend')
        and backend_map['yuanrong']['path'].endswith('yuanrong_backend.yuanrong_backend')
        and registry['AscendStoreConnector']['module'].endswith('store_utils.ascend_store_connector')
        and registry['MooncakeConnectorStoreV1']['module'].endswith('store_utils.ascend_store_connector')
        and registry['UCMConnector']['module'].endswith('ucm_connector.ucm_connector')
        and registry['LMCacheAscendConnector']['module'].endswith('lmcache_ascend_connector.lmcache_ascend_connector')
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result['ok']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
