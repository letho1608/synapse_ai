// Synapse AI — UI Store: ephemeral user interface state (not server data)
document.addEventListener('alpine:init', function () {
  Alpine.store('ui', {
    // Model filter state
    modelSearch: '',
    modelStatusFilter: 'all',

    // Test/Chat page state
    chatModel: '',
    chatMessages: [],
    chatInput: '',
    chatTemperature: 0.7,
    isChatting: false,
    abortController: null,

    // Training form state
    trainingModel: '',
    trainingDataset: '',
    trainingEpochs: 10,
    trainingBatchSize: 16,
    trainingLearningRate: '1e-5',
    trainingSaveEvery: 5,
    trainingMaxSteps: 0,
    trainingWarmupSteps: 0,
    trainingMaxLength: 128,
    trainingOutputName: '',
    trainingCheckpointDir: 'checkpoints',
    trainingSystemPrompt: '',
    // LoRA params (restored feature)
    loraRank: 8,
    loraAlpha: 16,
    loraDropout: 0.1,
    loraTargetModules: 'all',

    // Loading states (use object so Alpine can track per-key)
    loading: {},

    // Dataset modals
    datasetCreateOpen: false,
    datasetCreateName: '',
    datasetCreateFormat: 'jsonl',
    datasetCreateTemplate: 'empty',
    datasetCreateContent: '',
    datasetRenameOpen: false,
    datasetRenameTarget: null,
    datasetRenameNewName: '',
    datasetValidateResult: null,
    datasetPreview: [],
    datasetPreviewName: '',
    datasetPreviewLines: 50,
    datasetPreviewTarget: null,

    // Settings
    settingsLoading: false,

    // Notification system
    notifications: [],
    activeModal: null,

    // Computed helpers
    setLoading(key, val) { this.loading[key] = val; },
    getLoading(key) { return this.loading[key] || null; },

    allFilteredModels() {
      const store = Alpine.store('app');
      const search = (this.modelSearch || '').toLowerCase();
      const status = this.modelStatusFilter || 'all';
      const all = store.models.all;
      return all.filter(m => {
        const matchSearch = !search || (m.name || '').toLowerCase().includes(search) || (m.family || '').toLowerCase().includes(search);
        const matchStatus = status === 'all' || (status === 'downloaded' && m.isDownloaded) || (status === 'available' && !m.isDownloaded);
        return matchSearch && matchStatus;
      });
    }
  });
});