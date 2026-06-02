// Synapse AI — API Endpoint Constants
Synapse.API = Synapse.API || {};

Synapse.API.ENDPOINTS = {
  STATS:             '/v1/system/stats',
  SYSTEM_INFO:       '/v1/system/info',
  MODELS:            '/v1/models',
  MODEL_LIST:        '/v1/models/list',
  MODEL_COMPAT:      '/v1/models/compatibility',
  MODEL_STATUS:      '/v1/models/status',
  MODEL_PULL:        '/v1/models/pull',
  MODEL_DELETE:      '/v1/models/delete',
  DATASETS:          '/v1/datasets',
  DATASET_UPLOAD:    '/v1/datasets/upload',
  DATASET_CREATE:    '/v1/datasets/create',
  DATASET_DELETE:    '/v1/datasets/delete',
  DATASET_RENAME:    '/v1/datasets/rename',
  DATASET_VALIDATE:  '/v1/datasets/validate',
  DATASET_PREVIEW:   '/v1/datasets/preview',
  DATASET_DOWNLOAD:  '/v1/datasets/download',
  TRAINING_START:    '/v1/training/start',
  TRAINING_STATUS:   '/v1/training/status',
  CLUSTER_RESOURCES: '/v1/cluster/resources',
  TAILSCALE_NODES:   '/v1/tailscale/nodes',
  DISTRIBUTED_STATUS:'/v1/distributed/status',
  SETTINGS:          '/v1/settings',
  ACTIVITY:          '/v1/activity',
  LOGS_TERMINAL:     '/v1/logs/terminal',
  CHAT_COMPLETIONS:  '/v1/chat/completions'
};